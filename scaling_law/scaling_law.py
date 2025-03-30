import json
import glob
import numpy as np
import matplotlib.pyplot as plt
###########
### Save all.json loss files in folder scaling_law
### and run python scaling_law.py
########
# Load all loss JSON files from the directory
files = sorted(glob.glob('./*.json'))

# Prepare plot
plt.figure(figsize=(10, 7))
colors = plt.cm.tab10.colors
for idx, file in enumerate(files):
    with open(file, 'r') as f:
        data = json.load(f)

    steps = np.array(data['steps'])
    batch_size = data['batch_size']
    n_layer = data['n_layer']
    window_size = data['window_size']
    embd_dim = data['embd_dim']
    total_params = data['total_params']

    # Calculate number of computations (approximate)
    print("batch_size=",batch_size)
    print("n_layer=",n_layer)
    print("window_size=",window_size)
    print("embd_dim=",embd_dim)


    computations = steps * batch_size * n_layer *3* (11 * window_size * embd_dim + 2 * embd_dim*window_size**2 )
    computations_in_P = computations / (1e15*0.013*86400)

    train_losses = np.array(data['train_losses'])
    val_losses = np.array(data['val_losses'])

    # Format parameter count nicely (e.g., "12.34M")
    if total_params >= 1e6:
        params_label = f"{total_params / 1e6:.2f}M"
    else:
        params_label = f"{total_params / 1e3:.2f}K"

    label_train = f"Train loss ({params_label} params)"
    label_val = f"Val loss ({params_label} params)"


    color = colors[idx % len(colors)]  # consistent color for train/val pair
    plt.plot(computations_in_P, train_losses, color=color, linestyle='-', label=label_train)
    plt.plot(computations_in_P, val_losses, color=color,  linestyle='--', label=label_val)

C_ref = np.logspace(-6, 5, 1000)
L_ref = 2.57 * C_ref ** (-0.048)
plt.plot(C_ref, L_ref, color='black', linestyle=':', linewidth=2, label=r'$L=2.57 \times C^{-0.048}$')

#C_ref = np.arange(0, 10000, 0.01)
#L_ref = 2.57 * C_ref ** (-0.048)
#plt.plot(C_ref, L_ref, color='black', linestyle=':', linewidth=2, label=r'$L=2.57 \times C^{-0.048}$')

# Set plot to log-log scale
plt.xscale('log')
plt.yscale('log')
plt.xlim(-0.01,0.01)#(1e-6, 1e4)#1e4
#plt.ylim(1.5, 6) #
plt.xlabel('Computations (PetaFLOPs, log scale)')
plt.ylabel('Loss (log scale)')
plt.title('Scaling Law Analysis')
plt.legend()
plt.grid(True, which="both", linestyle="--", alpha=0.7)
plt.tight_layout()

# Save plot
plt.savefig('scaling_law_plot.png', dpi=300)

# Display plot
plt.show()
