import numpy as np
#from models import lane_mod
#from models import vel_mod
from models import pos_vel_x
from models import pos_vel_y

def imm_alg_inputs(nI, F, G, H, Q, R, Pi_ap, y, u, muj, xj, Pj):
    x = 0*xj[0]
    P = 0*Pj[0]
    muj_tilde = 0*Pi_ap
    xj0 = [0*x for _ in range(nI)] # combined state estimate
    Pj0 = [0*P for _ in range (nI)] # combined covariance matrix
    cjbar = []
    # state interaction
    for j in range(nI):
        cjbar.append(Pi_ap[:, j].T.dot(muj))
        for i in range(nI):
            muj_tilde[i, j] = muj[i]*Pi_ap[i, j]/cjbar[j]
            xj0[j] += xj[i]*muj_tilde[i, j]
        for i in range(nI):
            Pj0[j] += muj_tilde[i, j]*(Pj[i]+(xj[i]-xj0[j]).dot((xj[i]-xj0[j]).T))
    # model probability update
    lambdaj = np.zeros((nI, 1))
    for j in range(nI):
        xj_tilde = F[j].dot(xj0[j])+G[j].dot(u[j])
        Pj_tilde = F[j]*Pj0[j]*F[j].T+Q[j]
        z = y-H.dot(xj_tilde)
        Sj_tilde = H.dot(Pj_tilde).dot(H.T)+R
        Sj_tildeinv = np.linalg.inv(Sj_tilde)
        Lj = Pj_tilde.dot(H.T).dot(np.linalg.inv(Sj_tilde))
        xj[j] = xj_tilde+Lj.dot(z)
        Pj[j] = (np.eye(x.shape[0])-Lj.dot(H)).dot(Pj_tilde)
        lambdaj[j] = np.exp(-0.5*z.T.dot(Sj_tildeinv).dot(z))/np.sqrt(np.linalg.det(2*np.pi*Sj_tilde))
    const = lambdaj.T.dot(cjbar)
    # # state estimate combination
    # for j in range(nI):
    #     muj[j] = lambdaj[j]*cjbar[j]/const
    #     x += muj[j]*xj[j]
    # for j in range(nI):
    #     P += muj[j]*(Pj[j]+(x-xj[j]).dot((x-xj[j]).T))
    return muj, xj, Pj, x, P


nI = 3; #  number of candidate models
Pi_ap =  np.array([[0.7, 0.15, 0.15], [0.15, 0.7, 0.15], [0.15, 0.15, 0.7]]) #   a-priori switching probability
F = [np.random.rand(4, 4) for j in range(nI)]  # list of closed-loop matrices
G = [np.random.rand(4, 2) for j in range(nI)]
H = np.array([[1, 0, 0, 0], [0, 0, 1, 0]])  # this is not a list, is the same for all models
Q = [np.diag([0.1, 0.5, 0.1, 0.5]) for j in range(nI)] # process noise covariance
R = np.diag([0.05, 0.05]) # measurement noise covariance (this is not a list, is the same for all models)
u = [np.random.rand(2, 1) for j in range(nI)]  # list of closed-loop inputs (-K.dot(x_ref))
muj = np.ones((nI, 1))/nI   #   probabilities current estimation (initialized as equally likely)
xj = [np.array([[0.0, 0.0, 0.0, 0.0]]).T for j in range(nI)]    #   list of states for every model (internal variable of IMM, must be initialized with something reasonable before iteration 0)
Pj = [Q[0] for j in range(nI)] #   covariance matrix of estimate according to each model (internal variable of IMM)

Nsteps = 100
for k in range(Nsteps):
    y = np.random.rand(2, 1) # last measurement (updated at every sampling time)
    imm_output = imm_alg_inputs(nI, F, G, H, Q, R, Pi_ap, y, u, muj, xj, Pj)
    muj = imm_output[0]
    xj = imm_output[1]
    Pj = imm_output[2]