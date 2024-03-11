import haiku as hk
import jax
import optax
import jax.numpy as jnp
import argparse
import numpy as np
from tqdm import tqdm
from functools import partial
from jax.experimental.jet import jet
import os
import pandas as pd

parser = argparse.ArgumentParser(description='PINN Training')
parser.add_argument('--SEED', type=int, default=0)
parser.add_argument('--dim', type=int, default=100) # dimension of the problem.
parser.add_argument('--dataset', type=str, default="Sine_Gordon")
parser.add_argument('--epochs', type=int, default=20001) # Adam epochs
parser.add_argument('--lr', type=float, default=1e-3) # Adam lr
parser.add_argument('--PINN_h', type=int, default=128) # width of PINN
parser.add_argument('--PINN_L', type=int, default=4) # depth of PINN
parser.add_argument('--N_f', type=int, default=int(100)) # num of residual points
parser.add_argument('--N_test', type=int, default=int(20000)) # num of test points
parser.add_argument('--x_radius', type=float, default=1)
parser.add_argument('--algo', type=str, default='pinn')
parser.add_argument('--method', type=int, default=0)
parser.add_argument('--V', type=int, default=16)
parser.add_argument('--lam', type=float, default=1e-2)
parser.add_argument('--freq', type=float, default=1)
parser.add_argument('--save_loss', type=int, default=0)
args = parser.parse_args()
print(args)

np.random.seed(args.SEED)
key = jax.random.PRNGKey(args.SEED)
assert args.dataset == "Sine_Gordon"

const_2 = args.freq
c = np.random.randn(1, args.dim - 2)
def load_data_TwoBody_Sine_Gordon(d):
    args.input_dim = d
    args.output_dim = 1
    def func_u(x):
        x1, x2, x3 = x[:, :-2], x[:, 1:-1], x[:, 2:]
        temp =  args.x_radius**2 - np.sum(x**2, 1)
        temp2 = c * np.exp(x1 * x2 * x3)
        temp2 = np.sum(temp2, 1)
        return temp * temp2

    N_test = args.N_test

    x = np.random.randn(N_test, d)
    r = np.random.rand(N_test, 1) * args.x_radius
    x = x / np.linalg.norm(x, axis=1, keepdims=True) * r
    u = func_u(x)
    return x, u

x, u = load_data_TwoBody_Sine_Gordon(d=args.dim)
print(x.shape, u.shape)
print(u.mean(), u.std())

class MLP(hk.Module):
    def __init__(self, layers):
        super().__init__()
        self.layers = layers

    def __call__(self, x):
        # temp =  1 - jnp.sum(x**2)
        # temp2 = c * jnp.sin(x[:-1] + const_2 * jnp.cos(x[1:]) + x[1:] * jnp.cos(x[:-1]))
        # temp2 = jnp.sum(temp2)
        # return temp * temp2
        boundary_aug = args.x_radius**2 - jnp.sum(x**2)
        X = x
        for dim in self.layers[:-1]:
            X = hk.Linear(dim)(X)
            X = jnp.tanh(X)
        X = hk.Linear(self.layers[-1])(X)
        X = X[0]
        X = X * boundary_aug
        return X

class PINN:
    def __init__(self):
        self.epoch = args.epochs
        self.adam_lr = args.lr
        self.X, self.U = x, u

        layers = [args.PINN_h] * (args.PINN_L - 1) + [1]
        @hk.transform
        def network(x):
            temp = MLP(layers=layers)
            return temp(x)
 
        self.u_net = hk.without_apply_rng(network)
        self.u_pred_fn = jax.vmap(self.u_net.apply, (None, 0)) # consistent with the dataset
        self.r_pred_fn = jax.vmap(self.residual, (None, 0))
        self.r_gpinn_fn = jax.vmap(jax.grad(self.residual, argnums=1), (None, 0))
        self.r_rand_fn = jax.vmap(jax.vmap(self.residual_randomized, (None, 0, None)), (None, None, 0))
        # self.r_gpinn_rand_fn = jax.vmap(jax.vmap(self.gpinn_randomized, (None, 0, None)), (None, None, 0))
        self.r_gpinn_rand_fn = jax.vmap(jax.vmap(jax.grad(self.residual_randomized, argnums=1), \
                                                 (None, 0, None)), (None, None, 0))

        self.params = self.u_net.init(key, self.X[0])
        lr = optax.linear_schedule(
            init_value=self.adam_lr, end_value=0,
            transition_steps=args.epochs,
            transition_begin=0
        )
        self.optimizer = optax.adam(lr)
        self.opt_state = self.optimizer.init(self.params)

        self.saved_loss = []
        self.saved_l2 = []

    def resample(self, rng): # sample random points at the begining of each iteration
        keys = jax.random.split(rng, 3)

        N_f = args.N_f # Number of collocation points

        xf = jax.random.normal(keys[0], shape=(N_f, args.dim))
        rf = jax.random.uniform(keys[1], shape=(N_f, 1)) * args.x_radius
        xf = xf / jnp.linalg.norm(xf, axis=1, keepdims=True) * rf

        def residual_exact(x):
            u1 = args.x_radius**2 - np.sum(x**2)
            du1_dx = -2 * x
            d2u1_dx2 = -2

            x1, x2, x3 = x[:-2], x[1:-1], x[2:]
            coeffs = c.reshape(-1)
            u2 = coeffs * jnp.exp(x1 * x2 * x3)
            u2 = jnp.sum(u2)
            du2_dx_part = coeffs * jnp.exp(x1 * x2 * x3)
            du2_dx = jnp.zeros((args.dim,))
            du2_dx = du2_dx.at[:-2].add(du2_dx_part)
            du2_dx = du2_dx.at[1:-1].add(du2_dx_part)
            du2_dx = du2_dx.at[2:].add(du2_dx_part)
            d2u2_dx2 = du2_dx
            ff = u1 * d2u2_dx2 + 2 * du1_dx * du2_dx + u2 * d2u1_dx2
            ff = jnp.sum(ff)

            u = (u1 * u2)
            ff = ff + jnp.sin(u)

            return ff
        
        ff = (jax.vmap(residual_exact))(xf)
        ff_gPINN = (jax.vmap(jax.grad(residual_exact))(xf)) if args.algo == 'gpinn' else 0

        return xf, ff, ff_gPINN, keys[2]

    def residual(self, params, x): 
        u = self.u_net.apply(params, x)
        u_hess = jax.jacfwd(jax.jacrev(self.u_net.apply, argnums=1), argnums=1)(params, x)
        # u_hess = jax.hessian(self.u_net.apply, argnums=1)(params, x)
        u_xx = jnp.sum(jnp.diag(u_hess))
        return u_xx + jnp.sin(u)
    
    def residual_randomized(self, params, x, v):
        u = self.u_net.apply(params, x)
        f = lambda x: self.u_net.apply(params, x)
        _, (_, u_xx) = jet(f, (x, ), [[v, jnp.zeros_like(v)]]) #  Taylor-mode AD
        return u_xx + jnp.sin(u)
    
    def gpinn_randomized(self, params, x, v):
        f = lambda x: self.residual(params, x)
        return jax.jvp(f, (x, ), (v, ))[1]
    
    def get_loss_pinn(self, params, xf, ff):
        f = self.r_pred_fn(params, xf)
        mse_f = jnp.mean((f - ff)**2)
        return mse_f
    
    def get_loss_gpinn(self, params, xf, ff, ff_gPINN):
        f = self.r_pred_fn(params, xf)
        mse_f = jnp.mean((f - ff)**2)
        f2 = self.r_gpinn_fn(params, xf)
        mse_f2 = jnp.mean((f2 - ff_gPINN)**2)
        return mse_f + args.lam * mse_f2, (mse_f, mse_f2)
    
    def get_loss_rand_pinn(self, params, xf, ff, v):
        f = self.r_rand_fn(params, xf, v)
        f = f.mean(0)
        mse_f = jnp.mean((f - ff)**2)
        return mse_f
    
    def get_loss_rand_gpinn(self, params, xf, ff, v, ff_gPINN):
        f = self.r_rand_fn(params, xf, v)
        f = f.mean(0)
        mse_f = jnp.mean((f - ff)**2)
        f2 = self.r_gpinn_rand_fn(params, xf, v)
        f2 = f2.mean(0)
        mse_f2 = jnp.mean((f2 - ff_gPINN)**2)
        return mse_f + args.lam * mse_f2, (mse_f, mse_f2)
    
    @partial(jax.jit, static_argnums=(0,))
    def step_pinn(self, params, opt_state, rng):
        xf, ff, ff_gPINN, rng = self.resample(rng)
        current_loss, gradients = jax.value_and_grad(self.get_loss_pinn)(params, xf, ff)
        updates, opt_state = self.optimizer.update(gradients, opt_state)
        params = optax.apply_updates(params, updates)
        return current_loss, params, opt_state, rng
    
    @partial(jax.jit, static_argnums=(0,))
    def step_gpinn(self, params, opt_state, rng):
        xf, ff, ff_gPINN, rng = self.resample(rng)
        current_loss, gradients = jax.value_and_grad(self.get_loss_gpinn, has_aux=True)(params, xf, ff, ff_gPINN)
        updates, opt_state = self.optimizer.update(gradients, opt_state)
        params = optax.apply_updates(params, updates)
        return current_loss[1], params, opt_state, rng
    
    @partial(jax.jit, static_argnums=(0,))
    def step_rand_pinn(self, params, opt_state, rng):
        xf, ff, ff_gPINN, rng = self.resample(rng)
        keys = jax.random.split(rng, 2)
        # v = jax.random.normal(keys[0], shape=(args.V, args.dim))
        v = 2 * (jax.random.randint(keys[0], shape=(args.V, args.dim), minval=0, maxval=2) - 0.5)
        current_loss, gradients = jax.value_and_grad(self.get_loss_rand_pinn)(params, xf, ff, v)
        updates, opt_state = self.optimizer.update(gradients, opt_state)
        params = optax.apply_updates(params, updates)
        return current_loss, params, opt_state, keys[1]
    
    @partial(jax.jit, static_argnums=(0,))
    def step_rand_gpinn(self, params, opt_state, rng):
        xf, ff, ff_gPINN, rng = self.resample(rng)
        keys = jax.random.split(rng, 2)
        # v = jax.random.normal(keys[0], shape=(args.V, args.dim))
        v = 2 * (jax.random.randint(keys[0], shape=(args.V, args.dim), minval=0, maxval=2) - 0.5)
        current_loss, gradients = jax.value_and_grad(self.get_loss_rand_gpinn, has_aux=True)(params, xf, ff, v, ff_gPINN)
        updates, opt_state = self.optimizer.update(gradients, opt_state)
        params = optax.apply_updates(params, updates)
        return current_loss[1], params, opt_state, keys[1]

    def train_adam(self):
        self.rng = jax.random.PRNGKey(args.SEED)
        for n in tqdm(range(self.epoch)):
            if args.method == 0 and args.algo == 'pinn':
                current_loss, self.params, self.opt_state, self.rng = self.step_pinn(self.params, self.opt_state, self.rng)
                if args.save_loss: current_l2 = self.L2_pinn(self.params, self.X, self.U)
                if n%1000==0: print('epoch %d, loss: %e, L2: %e'%(n, current_loss, self.L2_pinn(self.params, self.X, self.U)))
            elif args.method == 1 and args.algo == 'pinn':
                current_loss, self.params, self.opt_state, self.rng = self.step_rand_pinn(self.params, self.opt_state, self.rng)
                if args.save_loss: current_l2 = self.L2_pinn(self.params, self.X, self.U)
                if n%1000==0: print('epoch %d, loss: %e, L2: %e'%(n, current_loss, self.L2_pinn(self.params, self.X, self.U)))
            elif args.method == 0 and args.algo == 'gpinn':
                current_loss, self.params, self.opt_state, self.rng = self.step_gpinn(self.params, self.opt_state, self.rng)
                if args.save_loss: current_l2 = self.L2_pinn(self.params, self.X, self.U)
                if n%1000==0: print('epoch %d, PINN/gPINN: %e/%e, L2: %e'%(n, current_loss[0], current_loss[1], self.L2_pinn(self.params, self.X, self.U)))
                current_loss = current_loss[0]
            elif args.method == 1 and args.algo == 'gpinn':
                current_loss, self.params, self.opt_state, self.rng = self.step_rand_gpinn(self.params, self.opt_state, self.rng)
                if args.save_loss: current_l2 = self.L2_pinn(self.params, self.X, self.U)
                if n%1000==0: print('epoch %d, PINN/gPINN: %e/%e, L2: %e'%(n, current_loss[0], current_loss[1], self.L2_pinn(self.params, self.X, self.U)))
                current_loss = current_loss[0]
            if args.save_loss: self.saved_loss.append(current_loss)
            if args.save_loss: self.saved_l2.append(current_l2)
    @partial(jax.jit, static_argnums=(0,)) 
    def L2_pinn(self, params, x, u):
        pinn_u_pred_20 = self.u_pred_fn(params, x).reshape(-1)
        pinn_error_u_total_20 = jnp.linalg.norm(u - pinn_u_pred_20, 2) / jnp.linalg.norm(u, 2)
        return (pinn_error_u_total_20)

model = PINN()
model.train_adam()
if not os.path.exists("records_hte"): os.makedirs("records_hte")
if args.save_loss:
    model.saved_loss = np.asarray(model.saved_loss)
    model.saved_l2 = np.asarray(model.saved_l2)

    info_dict = {"loss": model.saved_loss, "L2": model.saved_l2}
    df = pd.DataFrame(data=info_dict, index=None)
    df.to_excel(
        "records_hte/Sin_Gordon_D="+str(args.dim)+"_method="+str(args.method)+"_"+args.algo+"_V="+str(args.V)+"_S="+str(args.SEED)+".xlsx",
        index=False
    )
