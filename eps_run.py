import numpy as np
import subprocess


start = 0.16
end = 0.16
step = 0.02


for i in np.arange(start, end + step, step):
    subprocess.run(["python3", "/home/user/bob/fakeguard/guard.py", str(i)])

