import json
import numpy as np
import os
from typing import Dict, List, Any

def load_smplx_json(json_path: str) -> Dict[str, Any]:
    """
    SMPL-X JSON 결과 파일을 로드합니다.
    
    Args:
        json_path: JSON 파일 경로
        
    Returns:
        로드된 JSON 데이터
    """
    with open(json_path, 'r') as f:
        data = json.load(f)
    return data

def save_smplx_json(data: Dict[str, Any], json_path: str):
    """
    SMPL-X 결과를 JSON 파일로 저장합니다.
    
    Args:
        data: 저장할 데이터
        json_path: 저장할 파일 경로
    """
    with open(json_path, 'w') as f:
        json.dump(data, f, indent=2)

def extract_smplx_params(json_data: Dict[str, Any], person_id: int = 0) -> Dict[str, np.ndarray]:
    """
    JSON 데이터에서 SMPL-X 파라미터를 추출합니다.
    
    Args:
        json_data: JSON 데이터
        person_id: 추출할 사람의 ID (기본값: 0)
        
    Returns:
        SMPL-X 파라미터 딕셔너리
    """
    if not json_data['persons'] or person_id >= len(json_data['persons']):
        raise ValueError(f"Person ID {person_id} not found in JSON data")
    
    person = json_data['persons'][person_id]
    smplx_params = person['smplx_params']
    
    result = {
        'root_pose': np.array(smplx_params['root_pose']),
        'body_pose': np.array(smplx_params['body_pose']),
        'left_hand_pose': np.array(smplx_params['left_hand_pose']),
        'right_hand_pose': np.array(smplx_params['right_hand_pose']),
        'jaw_pose': np.array(smplx_params['jaw_pose']),
        'shape': np.array(smplx_params['shape']),
        'expression': np.array(smplx_params['expression']),
        'cam_trans': np.array(smplx_params['cam_trans']),
        'joints_3d': np.array(person['joints_3d']),
        'joints_2d': np.array(person['joints_2d']),
        'mesh_vertices': np.array(person['mesh_vertices']),
        'bbox': np.array(person['bbox'])
    }
    
    # 카메라 파라미터 추가 (프레임 레벨 또는 개인 레벨에서)
    if 'camera_params' in json_data:
        result['camera_params'] = json_data['camera_params']
    elif 'camera_params' in person:
        result['camera_params'] = person['camera_params']
    
    return result

def get_smplx_params_summary(json_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    JSON 데이터의 SMPL-X 파라미터 요약 정보를 반환합니다.
    
    Args:
        json_data: JSON 데이터
        
    Returns:
        요약 정보 딕셔너리
    """
    summary = {
        'frame_id': json_data['frame_id'],
        'image_path': json_data['image_path'],
        'image_size': json_data['image_size'],
        'num_persons': len(json_data['persons']),
        'persons': []
    }
    
    # 카메라 파라미터 추가
    if 'camera_params' in json_data:
        summary['camera_params'] = json_data['camera_params']
    
    for i, person in enumerate(json_data['persons']):
        person_summary = {
            'person_id': person['person_id'],
            'bbox': person['bbox'],
            'param_shapes': {
                'root_pose': len(person['smplx_params']['root_pose']),
                'body_pose': len(person['smplx_params']['body_pose']),
                'left_hand_pose': len(person['smplx_params']['left_hand_pose']),
                'right_hand_pose': len(person['smplx_params']['right_hand_pose']),
                'jaw_pose': len(person['smplx_params']['jaw_pose']),
                'shape': len(person['smplx_params']['shape']),
                'expression': len(person['smplx_params']['expression']),
                'cam_trans': len(person['smplx_params']['cam_trans'])
            },
            'joints_3d_shape': [len(person['joints_3d']), len(person['joints_3d'][0])],
            'joints_2d_shape': [len(person['joints_2d']), len(person['joints_2d'][0])],
            'mesh_vertices_shape': [len(person['mesh_vertices']), len(person['mesh_vertices'][0])]
        }
        summary['persons'].append(person_summary)
    
    return summary

def load_sequence_json(json_folder: str, start_frame: int = 1, end_frame: int = None) -> List[Dict[str, Any]]:
    """
    연속된 프레임의 JSON 파일들을 로드합니다.
    
    Args:
        json_folder: JSON 파일들이 있는 폴더
        start_frame: 시작 프레임 번호
        end_frame: 끝 프레임 번호 (None이면 모든 파일)
        
    Returns:
        JSON 데이터 리스트
    """
    json_files = sorted([f for f in os.listdir(json_folder) if f.endswith('.json')])
    
    if end_frame is None:
        end_frame = len(json_files)
    
    sequence_data = []
    for i in range(start_frame - 1, min(end_frame, len(json_files))):
        json_path = os.path.join(json_folder, json_files[i])
        data = load_smplx_json(json_path)
        sequence_data.append(data)
    
    return sequence_data

def print_smplx_params_info(json_data: Dict[str, Any]):
    """
    SMPL-X 파라미터 정보를 출력합니다.
    
    Args:
        json_data: JSON 데이터
    """
    summary = get_smplx_params_summary(json_data)
    
    print(f"Frame ID: {summary['frame_id']}")
    print(f"Image: {summary['image_path']}")
    print(f"Image Size: {summary['image_size']}")
    print(f"Number of Persons: {summary['num_persons']}")
    
    # 카메라 파라미터 출력
    if 'camera_params' in summary:
        print(f"Camera Parameters:")
        print(f"  Focal: {summary['camera_params']['focal']}")
        print(f"  Principal Point: {summary['camera_params']['princpt']}")
    
    for person in summary['persons']:
        print(f"\nPerson {person['person_id']}:")
        print(f"  BBox: {person['bbox']}")
        print(f"  Parameter Shapes:")
        for param_name, param_len in person['param_shapes'].items():
            print(f"    {param_name}: {param_len}")
        print(f"  Joints 3D: {person['joints_3d_shape']}")
        print(f"  Joints 2D: {person['joints_2d_shape']}")
        print(f"  Mesh Vertices: {person['mesh_vertices_shape']}")

if __name__ == "__main__":
    # 사용 예제
    import sys
    
    if len(sys.argv) > 1:
        json_path = sys.argv[1]
        data = load_smplx_json(json_path)
        print_smplx_params_info(data)
    else:
        print("사용법: python json_utils.py <json_file_path>") 