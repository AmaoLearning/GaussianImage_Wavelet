# import re
# import os
# from pathlib import Path

# # 正则模式
# pattern = re.compile(
#     r"Average:\s*(\d+)x(\d+),\s*"
#     r"PSNR:([\d.]+),\s*"
#     r"MS-SSIM:([\d.]+),\s*"
#     r"bpp:([\d.]+),\s*"
#     r"Eval:([\d.]+)s,\s*"
#     r"FPS:([\d.]+),\s*"
#     r"position_bpp:([\d.]+),\s*"
#     r"cholesky_bpp:([\d.]+),\s*"
#     r"feature_dc_bpp:([\d.]+)"
# )

# if __name__ == '__main__':

#     base_dir = '/ssd/huanxiong/gsimage/checkpoints_quant_v2/kodak' # GaussianImage_Cholesky_50000_1000
#     nums = ['800', '1000', '3000', '5000', '7000', '9000']

#     psnrs = []
#     bpps  = []

#     for num in nums:
#         # 读取 test.txt
#         file_path = Path(os.path.join(base_dir, f"GaussianImage_Cholesky_50000_{num}", "test.txt"))

#         with open(file_path, "r", encoding="utf-8") as f:
#             lines = [line.strip() for line in f if line.strip()]

#         # 取最后一行
#         last_line = lines[-1]
#         print("Last line:", last_line)

#         match = pattern.search(last_line)
#         if match:
#             data = {
#                 "width": int(match.group(1)),
#                 "height": int(match.group(2)),
#                 "psnr": float(match.group(3)),
#                 "ms_ssim": float(match.group(4)),
#                 "bpp": float(match.group(5)),
#                 "eval_time": float(match.group(6)),
#                 "fps": float(match.group(7)),
#                 "position_bpp": float(match.group(8)),
#                 "cholesky_bpp": float(match.group(9)),
#                 "feature_dc_bpp": float(match.group(10)),
#             }
#             print(data)
#         else:
#             print("No match found")

#         psnrs.append(data["psnr"])
#         bpps.append(data["bpp"])

#     print(psnrs)
#     print(bpps)


import re
import os
from pathlib import Path

# 修改正则模式，适配当前文本格式
pattern = re.compile(
    r"Average:\s*(\d+)x(\d+),\s*"
    r"PSNR:([\d.]+),\s*"
    r"MS-SSIM:([\d.]+),\s*"
    r"Training:([\d.]+)s,\s*"
    r"Eval:([\d.]+)s,\s*"
    r"FPS:([\d.]+)"
)

if __name__ == '__main__':

    base_dir = '/ssd/huanxiong/gsimage/checkpoints_v2/kodak'
    # nums = ['800', '1000', '3000', '5000', '7000', '9000']
    # nums = ['10000', '30000', '50000', '70000', '90000']
    nums = ['100000', '200000', '300000', '400000']

    psnrs = []

    for num in nums:
        file_path = Path(os.path.join(base_dir, f"GaussianImage_Cholesky_v2_1_50000_{num}", "train.txt"))

        with open(file_path, "r", encoding="utf-8") as f:
            lines = [line.strip() for line in f if line.strip()]

        last_line = lines[-1]
        print("Last line:", last_line)

        match = pattern.search(last_line)
        if match:
            data = {
                "width": int(match.group(1)),
                "height": int(match.group(2)),
                "psnr": float(match.group(3)),
                "ms_ssim": float(match.group(4)),
                "training_time": float(match.group(5)),
                "eval_time": float(match.group(6)),
                "fps": float(match.group(7)),
            }
            print(data)
            psnrs.append(data["psnr"])
        else:
            print("No match found")

    print(nums)
    print(psnrs)
