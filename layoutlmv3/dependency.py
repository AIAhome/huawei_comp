import json

# 修改该路径到你实际的requirements文件位置
req_path = "./model/requirements.txt"

data = []
with open(req_path, "r") as f:
    for line in f.readlines():
        if ">=" in line:
            restraint = "ATLEAST"
            d = ">="
        elif "<=" in line:
            restraint = "ATMOST"
            d = "<="
        else:
            restraint = "EXACT"
            d = "=="
        name, version = line.strip().split(d)
        data.append(dict(
            restraint=restraint,
            package_version=version,
            package_name=name
        ))

print(json.dumps(data, indent=4))