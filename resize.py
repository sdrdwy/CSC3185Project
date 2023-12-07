#How to use: python resize.py path width height
#Used for resizing image to smaller size
#Date: 2023/12/6
from PIL import Image
import os,sys
path=sys.argv[1]
size_w=int(sys.argv[2])
size_h=int(sys.argv[3])
for maindir,subdir,file_name_list in os.walk(path):
    length=len(file_name_list)
    cnt=0
    for file_name in file_name_list:
        cnt+=1
        image=os.path.join(maindir,file_name)
        if ".png" in image:
            file = Image.open(image)
            output= file.resize((size_w,size_h),Image.LANCZOS)
            output.save(image)
        print(f"\r Processing "+"%.2f"%((cnt/length)*100)+"%...",end='',flush=True)
print("\nDone.")