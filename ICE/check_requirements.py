import subprocess
import sys

# 필수 패키지 목록
required = [
    "torch",
    "torchvision",
    "Pillow",
    "matplotlib",
    "scikit-learn",
    "numpy",
    "tqdm"
]

# 설치된 패키지 목록 얻기
installed = subprocess.check_output([sys.executable, '-m', 'pip', 'freeze']).decode().lower()
installed_packages = [pkg.split('==')[0] for pkg in installed.splitlines()]

# 설치 안 된 패키지 찾기
missing = [pkg for pkg in required if pkg.lower() not in installed_packages]

# 결과 출력
if not missing:
    print("필수 패키지 설치 완료")
else:
    print("다음 필수 패키지들이 설치되어 있지 않음:\n")
    for pkg in missing:
        print(f"  - {pkg}")
    
    print("\n설치하려면 명령어를 입력:")
    print("pip install " + " ".join(missing))
