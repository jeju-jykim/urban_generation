import os
from openai import OpenAI
from dotenv import load_dotenv
import re

# .env 파일 로드
load_dotenv()

# OpenAI API 설정
client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))

TEMPLATE = """
Role: You serve as a Blender API operator to determine optimal building layout parameters based on physics analysis.
Task: Your task is to analyze the physics simulation report and determine the appropriate parameters for the "Building_Layout_Optimizer" API.

Document:
[{Name: Min_Spacing, Description: Minimum distance between building centers to ensure ventilation, Value: 2.5-10.0, Default: 2.5}],
[{Name: Seed, Description: Random seed for building placement pattern, Value: 0-100, Default: 0}],

Format: Output the modified parameters in the following form:
Min_Spacing: You need to give a parameter value in the range [2.5, 15.0], present as a float.
Seed: Integer value in range [0, 100], present as an integer.

Examples: 
Input: "Site shows 38°C surface temperature with 0.5m/s wind velocity. Poor ventilation detected."
Output: [Min_Spacing: 5.0, Seed: 0]

"""

USER_PROMPT = """
Question: {physics_description}

Current layout parameters:
- Min_Spacing: 2.5 (very dense)
- Seed: 0

Solution: Based on the physics report, determine the optimal Min_Spacing value to improve thermal comfort and ventilation.
"""

CODE_TEMPLATE = """
'''python
import bpy

def apply_building_layout(Min_Spacing, Seed):
    # Get the modifier
    obj = bpy.context.active_object
    mod = obj.modifiers["BuildingLayout"]
    
    # Apply parameters from LLM
    mod["Input_1"] = Min_Spacing  # Min Spacing
    mod["Input_2"] = Seed          # Seed
    
    bpy.context.view_layer.update()
    return "Layout updated"

# Execute with LLM parameters
apply_building_layout(Min_Spacing={min_spacing}, Seed={seed})
'''
"""

FULL_EXAMPLE = """
Question: Site Analysis Report shows:
- Surface temperature: 38°C at ground level
- Air temperature: 35°C at pedestrian level  
- Wind velocity: 0.5 m/s (nearly stagnant)
- Air exchange rate: 0.2 ACH (very poor)
- Thermal comfort: Unacceptable
- Recommendation: URGENT - Improve ventilation corridors

Solution: Based on the thermal stress and stagnant air conditions, the parameters should be adjusted to create wind corridors:

[Min_Spacing: 6.5, Seed: 0]

This increased spacing (from 2.5 to 6.5) will:
- Create ventilation corridors between buildings
- Reduce heat accumulation by 40%
- Increase wind velocity to approximately 2.8 m/s
- Improve air exchange rate to acceptable levels

Code:
'''python
apply_building_layout(Min_Spacing=6.5, Seed=0)
'''
"""

def get_parameters_from_physics(physics_description):
    prompt = TEMPLATE + "\n\n" + USER_PROMPT.format(
        physics_description=physics_description
    )
    
    try:
        print(f"\n🔍 Physics Analysis: {physics_description}")
        print("🤖 Calling OpenAI API for parameter optimization...")
        
        completion = client.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3,  # 낮은 temperature로 일관된 결과
            max_tokens=500
        )
        
        response = completion.choices[0].message.content.strip()
        print(f"📋 LLM Response: {response}")
        
        # 파라미터 추출
        min_spacing = None
        seed = None
        
        # Min_Spacing 추출
        if match := re.search(r"Min_Spacing:\s*([0-9.]+)", response):
            min_spacing = float(match.group(1))
        
        # Seed 추출  
        if match := re.search(r"Seed:\s*([0-9]+)", response):
            seed = int(match.group(1))
        
        # 값 검증
        if min_spacing is None or seed is None:
            print("❌ Failed to extract parameters from LLM response")
            return None, None
            
        if not (2.5 <= min_spacing <= 10.0):
            print(f"⚠️ Min_Spacing out of range: {min_spacing}, using default 5.0")
            min_spacing = 5.0
            
        if not (0 <= seed <= 100):
            print(f"⚠️ Seed out of range: {seed}, using default 0")
            seed = 0
        
        print(f"✅ Extracted Parameters:")
        print(f"   - Min_Spacing: {min_spacing}")
        print(f"   - Seed: {seed}")
        
        return min_spacing, seed
        
    except Exception as e:
        print(f"❌ OpenAI API Error: {str(e)}")
        return None, None

def test_llm_call():
    """테스트 함수"""
    test_physics = """
    Site Analysis Report shows:
    - Surface temperature: 38°C at ground level
    - Air temperature: 35°C at pedestrian level  
    - Wind velocity: 0.5 m/s (nearly stagnant)
    - Air exchange rate: 0.2 ACH (very poor)
    - Thermal comfort: Unacceptable
    - Recommendation: URGENT - Improve ventilation corridors
    """
    
    min_spacing, seed = get_parameters_from_physics(test_physics)
    
    if min_spacing is not None and seed is not None:
        print(f"\n🎯 Test Result:")
        print(f"   - Min_Spacing: {min_spacing}")
        print(f"   - Seed: {seed}")
        
        # Blender 코드 생성
        code = CODE_TEMPLATE.format(
            min_spacing=min_spacing,
            seed=seed
        )
        print(f"\n📝 Generated Blender Code:")
        print(code)
    else:
        print("❌ Test failed - no parameters extracted")

if __name__ == "__main__":
    test_llm_call()