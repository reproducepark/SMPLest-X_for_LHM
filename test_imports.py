#!/usr/bin/env python3
"""
모듈 임포트 테스트 스크립트
"""

import sys
import os
from pathlib import Path

# 프로젝트 루트 디렉토리를 Python 경로에 추가
root_dir = Path(__file__).resolve().parent
sys.path.insert(0, str(root_dir))

def test_imports():
    """모든 필요한 모듈들이 임포트되는지 테스트"""
    
    print("모듈 임포트 테스트 시작...")
    
    try:
        print("1. human_models 임포트 테스트...")
        from human_models.human_models import SMPL, SMPLX
        print("   ✓ human_models 성공")
    except Exception as e:
        print(f"   ✗ human_models 실패: {e}")
        return False
    
    try:
        print("2. utils 임포트 테스트...")
        from utils.data_utils import load_img, process_bbox
        from utils.inference_utils import non_max_suppression
        print("   ✓ utils 성공")
    except Exception as e:
        print(f"   ✗ utils 실패: {e}")
        return False
    
    try:
        print("3. main 임포트 테스트...")
        from main.base import Tester
        from main.config import Config
        print("   ✓ main 성공")
    except Exception as e:
        print(f"   ✗ main 실패: {e}")
        return False
    
    try:
        print("4. models 임포트 테스트...")
        from models.SMPLest_X import get_model
        print("   ✓ models 성공")
    except Exception as e:
        print(f"   ✗ models 실패: {e}")
        return False
    
    try:
        print("5. datasets 임포트 테스트...")
        from datasets.dataset import MultipleDatasets
        print("   ✓ datasets 성공")
    except Exception as e:
        print(f"   ✗ datasets 실패: {e}")
        return False
    
    print("\n모든 모듈 임포트 성공! ✓")
    return True

if __name__ == "__main__":
    success = test_imports()
    if not success:
        sys.exit(1) 