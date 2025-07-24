import json

# 读取原始JSON文件
with open('/home/rainbus/Projects/Other/Python/KgCoOp/KgCoOp/caltech101.json', 'r') as f:
    data = json.load(f)

# 转换数据，只保留每个列表中的第一个元素
transformed_data = {key: value[0] for key, value in data.items()}

# 将转换后的数据写回文件（你也可以选择写入新文件）
with open('/home/rainbus/Projects/Other/Python/KgCoOp/KgCoOp/caltech101_transformed.json', 'w') as f:
    json.dump(transformed_data, f, indent=4)
