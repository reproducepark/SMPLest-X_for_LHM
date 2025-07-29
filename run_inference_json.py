#!/usr/bin/env python3
"""
SMPL-X 추론을 실행하고 결과를 JSON으로 저장하는 스크립트
"""

import os
import sys
import subprocess
import argparse

# 현재 디렉토리를 Python 경로에 추가
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_dir)

def main():
    parser = argparse.ArgumentParser(description='SMPL-X 추론 실행 및 JSON 결과 저장')
    parser.add_argument('--input_folder', type=str, required=True, 
                       help='입력 이미지 폴더 경로 (demo/input_frames/폴더명)')
    parser.add_argument('--ckpt_name', type=str, default='model_dump',
                       help='체크포인트 이름 (기본값: model_dump)')
    parser.add_argument('--start_frame', type=int, default=1,
                       help='시작 프레임 번호 (기본값: 1)')
    parser.add_argument('--end_frame', type=int, default=1,
                       help='끝 프레임 번호 (기본값: 1)')
    parser.add_argument('--multi_person', action='store_true',
                       help='다중 인물 감지 활성화')
    
    args = parser.parse_args()
    
    # 입력 폴더에서 폴더명 추출
    folder_name = os.path.basename(args.input_folder)
    
    # 명령어 구성
    cmd = [
        'python', 'main/inference.py',
        '--file_name', folder_name,
        '--ckpt_name', args.ckpt_name,
        '--start', str(args.start_frame),
        '--end', str(args.end_frame),
        '--save_json'
    ]
    
    if args.multi_person:
        cmd.append('--multi_person')
    
    print(f"실행 명령어: {' '.join(cmd)}")
    print(f"입력 폴더: {args.input_folder}")
    print(f"JSON 결과 저장 위치: demo/output_json/{folder_name}/")
    
    # 명령어 실행
    try:
        subprocess.run(cmd, check=True)
        print("추론 완료!")
        print(f"JSON 파일들이 demo/output_json/{folder_name}/ 폴더에 저장되었습니다.")
    except subprocess.CalledProcessError as e:
        print(f"오류 발생: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main() 