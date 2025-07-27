import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse import diags, kron, identity
from scipy.sparse.linalg import spsolve
from scipy.sparse import block_diag
from scipy.interpolate import interp1d, RectBivariateSpline
from skimage.restoration import denoise_tv_chambolle
from skimage.data import shepp_logan_phantom
from skimage.transform import resize



# ---------- HENYEY-GREENSTEIN MODEL ----------
def henyey_greenstein(cos_theta, g):
    return (1 - g**2) / (1 + g**2 - 2 * g * cos_theta)**(3/2)

def compute_directional_gradients_2d(nx, ny, dx, dy):
    # Total number of grid nodes
    N = nx * ny

    # --- Main directions ---
    # Right (M_x)
    main_diag_x = np.ones(N)
    lower_diag_x = -np.ones(N - 1)
    lower_diag_x[np.arange(1, N) % nx == 0] = 0
    M_x = diags([main_diag_x, lower_diag_x], [0, -1], format='csr') / dx

    # Left (M_-x)
    main_diag_neg_x = np.ones(N)
    upper_diag_x = -np.ones(N - 1)
    upper_diag_x[np.arange(N - 1) % nx == (nx - 1)] = 0
    M_neg_x = diags([main_diag_neg_x, upper_diag_x], [0, 1], format='csr') / dx
    
    # Up (M_y)
    main_diag_y = np.ones(N)
    lower_diag_y = -np.ones(N - ny)
    M_y = diags([main_diag_y, lower_diag_y], [0, -nx], format='csr') / dy

    # Down (M_-y)
    main_diag_neg_y = np.ones(N)
    upper_diag_y = -np.ones(N - nx)
    M_neg_y = diags([main_diag_neg_y, upper_diag_y], [0, nx], format='csr') / dy

    # --- Diagonal directions ---
    M_diag_ur = (M_x + M_y) / np.sqrt(2) # Up-Right (1,1)/√2
    M_diag_ul = (M_neg_x + M_y) / np.sqrt(2) # Up-Left (-1,1)/√2
    M_diag_dl = (M_neg_x + M_neg_y) / np.sqrt(2) # Down-Left (-1,-1)/√2
    M_diag_dr = (M_x + M_neg_y) / np.sqrt(2) # Down-Right (1,-1)/√2 


    return M_x, M_diag_ur, M_y, M_diag_ul, M_neg_x, M_diag_dl, M_neg_y, M_diag_dr

# --- Simulation parameters ---
nx, ny = 64, 64
Lx, Ly = 2.0, 2.0
dx, dy = Lx / (nx - 1), Ly / (ny - 1)
scattering_coeff = 50
g = 0.85

# --- Directions encoded using basis (+x, -x, +y, -y) ---
directions = np.array([
    [1, 0, 0, 0],  # (1, 0)
    [1 / np.sqrt(2), 0, 1 / np.sqrt(2), 0], # (1/√2, 0, 1/√2, 0)
    [0, 0, 1, 0], # (0, 1)
    [0, 1 / np.sqrt(2), 1 / np.sqrt(2), 0], # (0, 1/√2, 1/√2, 0)
    [0, 1, 0, 0], # (0, -1)
    [0, 1 / np.sqrt(2), 0, 1 / np.sqrt(2)], # (0, 1/√2, 0, 1/√2)
    [0, 0, 0, 1], # (-1, 0)
    [1 / np.sqrt(2), 0, 0, 1 / np.sqrt(2)],  # (1/√2, 0, 0, 1/√2)
])

direction_new = np.array([
    [1, 0], 
    [1 / np.sqrt(2), 1 / np.sqrt(2)],
    [0, 1], 
    [-1 / np.sqrt(2), 1 / np.sqrt(2)],
    [-1, 0], 
    [-1 / np.sqrt(2), -1 / np.sqrt(2)],
    [0, -1], 
    [1 / np.sqrt(2), -1 / np.sqrt(2)],
])
n_dirs = direction_new.shape[0]

k_matrix = np.zeros((n_dirs, n_dirs))
for i in range(n_dirs):
    for j in range(n_dirs):
        cos_theta = np.dot(direction_new[i], direction_new[j])
        k_matrix[i, j] = henyey_greenstein(cos_theta, g)
k_matrix /= np.sum(k_matrix, axis=1, keepdims=True)

# --- Create spatial grid ---
x = np.linspace(-Lx / 2, Lx / 2, nx)
y = np.linspace(-Ly / 2, Ly / 2, ny)
X, Y = np.meshgrid(x, y)

# --- Light source: uniform value along a horizontal line at Y = 0.85 ---
y0 = 0.85
tolerance = dy / 2

q = np.zeros_like(X)
q[np.abs(Y - y0) < tolerance] = 50


# --- Get directional gradient operators ---
A_x, A_diag_ur, A_y, A_diag_ul, A_neg_x, A_diag_dl, A_neg_y, A_diag_dr = compute_directional_gradients_2d(nx, ny, dx, dy)

# --- Transport operators for each direction ---
N = nx * ny
A_directions = []
for vx, v_negx, vy, v_negy in directions:
    A_dir = (
        kron(diags([vx], [0]), A_x) +
        kron(diags([v_negx], [0]), A_neg_x) +
        kron(diags([vy], [0]), A_y) +
        kron(diags([v_negy], [0]), A_neg_y)
    )
    A_directions.append(A_dir)
A_directional = block_diag(A_directions, format='csr')


# --- Use Shepp-Logan phantom as absorption map ---
phantom = shepp_logan_phantom()
phantom_small = resize(phantom, (32, 32), mode='reflect', anti_aliasing=True)

phantom_full = np.zeros((ny, nx))
start_x = (nx - 32) // 2
start_y = (ny - 32) // 2
shift_up = 6
#phantom_full[start_y - shift_up : start_y - shift_up + 32, start_x : start_x + 32] = phantom_small
phantom_full[start_y:start_y+32, start_x:start_x+32] = phantom_small

# phantom_normalized = 20 * phantom_resized / phantom_resized.max()
phantom_normalized = 50 * phantom_full / phantom_full.max()

border = 5 

phantom_normalized[:border, :] = 0
phantom_normalized[-border:, :] = 0
phantom_normalized[:, :border] = 0 
phantom_normalized[:, -border:] = 0 

abs_coeff_map = phantom_normalized
absorption_coeff_vector = abs_coeff_map.flatten()


plt.figure(figsize=(6, 5))
plt.imshow(phantom, origin='lower', cmap='hot')
plt.title('phantom')
plt.colorbar()
plt.tight_layout()
plt.show()

# Combine all directional transport matrices
A_abs_scatter = kron(identity(n_dirs), diags([absorption_coeff_vector + scattering_coeff], [0], shape=(N, N)))

# --- Scattering redistribution operator ---
A_scatter = kron(k_matrix, diags([scattering_coeff], [0], shape=(N, N)))

# --- Final system matrix ---
A_total = A_directional + A_abs_scatter - A_scatter

def make_directional_source(q_base, direction_index, n_dirs):
    """
    Creates a source vector where the light is injected only in one direction.
    """
    q_vector = np.zeros((n_dirs, *q_base.shape))
    q_vector[direction_index] = q_base
    return q_vector.reshape(-1)

# Solve for one direction
q_vector = make_directional_source(q, direction_index=6, n_dirs=n_dirs)
Phi_flat = spsolve(A_total, q_vector)
Phi = Phi_flat.reshape((n_dirs, ny, nx))
Phi_total = Phi.sum(axis=0) * 2 * np.pi / 8


# --- ABSORBED ENERGY ---
absorption_map = absorption_coeff_vector.reshape(Phi_total.shape)
H_photoacoustic = Phi_total * absorption_map 


# ---------- VISUALIZATION ----------
plt.figure(figsize=(8, 6))
plt.imshow(Phi_total, extent=(-Lx/2, Lx/2, -Ly/2, Ly/2), origin='lower', cmap='hot')
plt.title('Fluence Φ_total (Transport DOM)')
plt.colorbar()
plt.show()

plt.figure(figsize=(6, 5))
plt.imshow(H_photoacoustic, extent=(-Lx/2, Lx/2, -Ly/2, Ly/2), origin='lower', cmap='hot')
plt.title('(μa * Φ)')
plt.colorbar()
plt.tight_layout()
plt.show()


# Constructs the kernel matrix used in the wave propagation process
def kernel(tV):
    N = len(tV)
    Amat = np.zeros((N, N))
    for i in range(N):
        for j in range(i):
            Amat[i, j] = np.sqrt(tV[i]**2 - tV[j]**2) - np.sqrt(tV[i]**2 - tV[j+1]**2)
    return Amat


# Performs wave star backprojection reconstruction
def wavstarprop(data, xV, tV, R0, T0):
    Nt, Nphi = data.shape
    Nx = len(xV)
    dt = tV[1] - tV[0]
    dphi = 2 * np.pi / Nphi
    phiV = np.linspace(0, 2 * np.pi, Nphi, endpoint=False)
    X, Y = np.meshgrid(xV, xV)
    MF = np.zeros((Nphi, Nt))
    Amat = kernel(tV)
    F1 = np.zeros((Nx, Nx))

    # Time filtering of the data using the kernel matrix
    for i in range(Nphi):
        p = data[:, i]
        p = np.gradient(p, dt)
        mf = Amat.T @ p
        tVmod = np.maximum(tV, 0.5 * dt)
        mf /= tVmod
        MF[i, :] = mf

    # Backprojection step
    for i in range(Nphi):
        phi = phiV[i]
        xc = R0 * np.cos(phi)
        yc = R0 * np.sin(phi)
        D = np.sqrt((X - xc)**2 + (Y - yc)**2)
        mf = MF[i, :]
        interp_func = interp1d(tV, mf, kind='linear', bounds_error=False, fill_value=0.0)
        fadd = -interp_func(D.ravel()).reshape(D.shape)
        F1 += fadd * dphi

    RR = np.sqrt(X**2 + Y**2)
    # WBP = np.rot90(F1, k = -1)
    WBP = (F1.T)/ np.pi
    WBP[RR > R0] = 0
    return WBP


# Simulates forward wave propagation from an initial condition
def wavprop(F , xV, tV, Nphi, R0):
    Xi = xV[-1]
    Xo = 2 * Xi
    Nx = len(xV)
    Nt = len(tV)
    dt = tV[1]-tV[0]
    Nxo = 2 * Nx
    xVo = np.linspace(-Xo, Xo, Nxo)
    phiV = np.linspace(0, 2 * np.pi, Nphi, endpoint=False)
    XC = R0 * np.cos(phiV)
    YC = R0 * np.sin(phiV)
    X, Y = np.meshgrid(xVo, xVo)

    # Frequency vectors for Fourier transform
    kV = np.fft.ifftshift(np.linspace(-Nxo/2, Nxo/2-1, Nxo)) * np.pi / Xo
    k1, k2 = np.meshgrid(kV, kV)
    km = np.sqrt(k1**2 + k2**2)

    Data = np.zeros((Nt, Nphi))
    F = np.pad(F, ((Nx//2, Nx//2), (Nx//2, Nx//2)), mode='constant')
    W = F.copy()
    Walt = np.zeros_like(W)
    V = W

    # Time-stepping loop using finite difference in time and FFT in space
    for ii in range(Nt):
        FV = np.fft.fft2(V)
        FW = np.fft.fft2(W)
        LW = -4 * np.sin(dt * km / 2)**2 * (FW - FV)
        LW1 = np.fft.ifft2(LW)
        Wneu = 2 * W - Walt + LW1
        Walt = W
        f = RectBivariateSpline(xVo, xVo, np.real(W))
        Pcirc_time = f(XC, YC, grid=False)
        Data[ii, :] = Pcirc_time
        W = Wneu
    return Data, np.real(W)


# --- Simulation parameters ---
Nt = 300
T0 = 2.2
tV = np.linspace(0, T0, Nt)
Nphi = 100
R0 = 1.0
Nx = 128
xV = np.linspace(-1, 1, Nx)

Phi_interp = RectBivariateSpline(x, y, H_photoacoustic.T)
F = Phi_interp(xV, xV)

Data, W = wavprop(F, xV, tV, Nphi, R0)
recon = wavstarprop(Data, xV, tV, R0, T0)


# --- Data ---
plt.subplot(1, 3, 1)
plt.imshow(Data, cmap='hot', aspect='auto')
plt.title("Data (time × detector)")
plt.xlabel("Detector index (φ)")
plt.ylabel("Time step")

# --- Source ---
plt.subplot(1, 3, 2)
plt.imshow(F, cmap='hot', extent=[-1, 1, -1, 1])
plt.title("True Source F")


# --- Wave field ---
plt.subplot(1, 3, 3)
plt.imshow(W, cmap='hot', extent=[-2, 2, -2, 2])
plt.title("Wavefield P (last step)")

# ---------- Plot ----------
plt.figure(figsize=(10, 5))
plt.subplot(1, 3, 1)
plt.imshow(F, extent=[-1.2,1.2,-1.2,1.2], cmap='hot')
plt.colorbar()
plt.title("True Source")

plt.subplot(1, 3, 2)
plt.imshow(recon, extent=[-1.2,1.2,-1.2,1.2], cmap='hot')
plt.colorbar()
plt.title("Reconstructed")

plt.subplot(1, 3, 3)
plt.imshow(np.abs(F - recon), extent=[-1.2,1.2,-1.2,1.2], cmap='hot')
plt.colorbar()
plt.title("Abs Error")

plt.tight_layout()
plt.show()

# --- TV projection Q2: Total Variation denoising ---
def Q2_TV_denoise(a_vector, nx, ny, weight=0.1):
    a_image = a_vector.reshape(ny, nx)
    a_denoised = denoise_tv_chambolle(a_image, weight=weight, channel_axis=None)
    return a_denoised.flatten()

# --- Q1 projection: Project absorption onto piecewise-constant block space ---
def Q1_piecewise(a, nx, ny, block_size=4):
    a_proj = np.copy(a.reshape(ny, nx))
    for i in range(0, ny, block_size):
        for j in range(0, nx, block_size):
            block = a_proj[i:i+block_size, j:j+block_size]
            avg = np.mean(block)
            a_proj[i:i+block_size, j:j+block_size] = avg
    return a_proj.flatten()


def huber_loss(residual, delta=1.0):
    abs_r = np.abs(residual)
    return np.sum(np.where(abs_r <= delta,
                           0.5 * residual**2,
                           delta * (abs_r - 0.5 * delta)))

def huber_gradient(residual, delta=1.0):
    return np.where(np.abs(residual) <= delta,
                    residual,
                    delta * np.sign(residual))


# --- Forward solve: Compute fluence for given absorption vector ---
def solve_fluence(a_vector):
    A_abs_scatter = kron(identity(n_dirs), diags([a_vector + scattering_coeff], [0], shape=(N, N)))
    A_total = A_directional + A_abs_scatter - A_scatter

    Phi_flat = spsolve(A_total, q_vector)
    Phi = Phi_flat.reshape((n_dirs, ny, nx))
    Phi_total = Phi.sum(axis=0) * 2 * np.pi / n_dirs
    return Phi_total, Phi

# --- Objective function (loss) ---
def compute_loss(a_vector, pl_s1, delta=1.0):
    Phi_total, _ = solve_fluence(a_vector)
    H = a_vector.reshape(ny, nx) * Phi_total
    H_interp = RectBivariateSpline(x, y, H.T)(xV, xV)
    pl_sim, _ = wavprop(H_interp, xV, tV, Nphi, R0)

    residual = pl_sim - pl_s1
    return huber_loss(residual.flatten(), delta)


def compute_adjoint_gradient_acoustic(a_vector, pl_target, delta=1.0):
    dt = tV[1] - tV[0]

    Phi_total, _ = solve_fluence(a_vector)
    H = a_vector.reshape(ny, nx) * Phi_total


    H_interp = RectBivariateSpline(x, y, H.T)(xV, xV)
    pl_sim, _ = wavprop(H_interp, xV, tV, Nphi, R0)

    residual = pl_sim - pl_target
    grad_huber = huber_gradient(residual.flatten(), delta).reshape(pl_sim.shape)

    backproj = wavstarprop(grad_huber, xV, tV, R0, T0)
    recon_interp = RectBivariateSpline(xV, xV, backproj.T)
    recon_xy = recon_interp(x, y)

    grad = Phi_total * recon_xy * dt
    return grad.flatten()

# --- Gradient descent optimization ---
def optimize_absorption(a_init, y_target, n_iter, alpha, block_size, delta = 1.0):
    a = a_init.copy()
    loss_history = []

    for it in range(n_iter):
        grad = compute_adjoint_gradient_acoustic(a, y_target, delta)
        a_new = a + alpha * grad
        a_new = np.maximum(a_new, 0)


        # a_q1 = Q1_piecewise(a_new, nx, ny, block_size)   # Q1: block averaging projection
        # a = Q2_TV_denoise(a_q1, nx, ny, weight=0.01)     # Q2: TV denoising
        # a = Q1_piecewise(a_new, nx, ny, block_size)
        
        a = a_new

        loss = compute_loss(a, y_target, delta)
        loss_history.append(loss)
        print(f"Iter {it+1:02d}: Loss = {loss:.6e}")

        
        if it > 0 and abs(loss_history[-1] - loss_history[-2]) < 1e-6:
            print(f"Early stopping at iteration {it+1}")
            break
        
    return a, loss_history 

# Ground truth absorption map
a_true = absorption_coeff_vector.copy()

# Generate synthetic target data (pressure field)
Phi_total_true, _ = solve_fluence(a_true)
F_true = a_true.reshape(ny, nx) * Phi_total_true
Phi_interp = RectBivariateSpline(x, y, F_true.T)
F_interp = Phi_interp(xV, xV)
pl_s1, _ = wavprop(F_interp, xV, tV, Nphi, R0)


Nt, Nphi = pl_s1.shape
dt = tV[1] - tV[0]
for i in range(Nphi):
    p = pl_s1[:, i]
    p = np.gradient(p, dt)
    pl_s1[:, i] = p

plt.plot(1, 1)
plt.imshow(pl_s1, cmap='viridis')
plt.title("Gradient")
plt.colorbar()

a_init = 0.1 * np.ones_like(a_true)

# --- Optimization ---
a_rec, losses = optimize_absorption(a_init, pl_s1, n_iter=10, alpha= 0.1 , block_size=2, delta=100)
a_rec = a_rec * np.max(a_true) / np.max(a_rec)

# ---------- Visualization ----------

plt.figure(figsize=(16, 5))

plt.subplot(1, 3, 1)
plt.imshow(a_true.reshape(ny, nx), cmap='viridis')
plt.title("Ground truth $\mu_a$")
plt.colorbar()

plt.subplot(1, 3, 2)
plt.imshow(a_rec.reshape(ny, nx), cmap='viridis')
plt.title("Reconstructed $\mu_a$")
plt.colorbar()

plt.subplot(1, 3, 3)
plt.plot(losses, marker='o')
plt.title("Loss over iterations")
plt.xlabel("Iteration")
plt.ylabel("Loss")
plt.grid()

plt.tight_layout()
plt.show()
