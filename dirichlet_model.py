import pymc3 as pm

##################################################################
# Prepare our mock data for our Dirichlet model
##################################################################
T1 = np.array([50, 20, 30, 40, 100, 200, 30, 40, 50, 70])
T2 = np.array([10,  5, 60, 10,  30,  40, 20, 10,100, 30])
T = np.vstack([T1,T2]).T
epsilon = 1

def make_noise(n, sd):
    model = pm.Model()
    with model:
        mu1 = pm.Normal("mu1", mu=0, sd=sd, shape=1)
    with model:
        step = pm.NUTS()
        trace = pm.sample(n, tune=1000, init=None, step=step, njobs=1)
    values = trace.get_values(trace.varnames[0])
    return(values)

epsilon = make_noise(T.shape[0], sd)
pi1 = np.array([0.3, 0.7])
K = T.dot(pi1) + epsilon.T


with Model() as dirichlet_model:
    # Prior distributions for latent variables
    pi = pm.Dirichlet(name="pi",theta=[1/2]*2)
