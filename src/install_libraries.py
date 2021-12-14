import sys
import subprocess

res = subprocess.run(['pip', 'install', 'category_encoders'], stdout=subprocess.PIPE)
sys.stdout.write(res.stdout)
res = subprocess.run(['pip', 'install', '-U', 'ta'], stdout=subprocess.PIPE)
sys.stdout.write(res.stdout)
