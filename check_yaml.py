import yaml

with open('data.yaml', 'r', encoding='utf-8') as f:
    content = f.read()

try:
    data = yaml.safe_load(content)
    print("YAML valid!")
    print("Data:", data)
except yaml.YAMLError as e:
    print("Error:", e)