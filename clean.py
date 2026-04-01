filename = "data/generic/FB.csv"
final = ""

with open(filename, "r") as file:
    content = file.read()

lines = [l for l in content.split("\n") if l.strip() != ""]
header = lines[0].split(",")
data_lines = lines[1:]

feature_names = {i: name for i, name in enumerate(header)}
feature_blacklist = ["Close_forcast"]

# Detect non-numeric columns to blacklist (except Close_forcast)
for line in data_lines:
    cols = line.split(",")
    for index, feature in enumerate(cols):
        name = feature_names[index]
        if name in feature_blacklist:
            continue
        try:
            float(feature)
        except:
            if name not in feature_blacklist:
                feature_blacklist.append(name)

# Build new CSV
# Header
new_header = ",".join(
    name for name in header if name not in feature_blacklist
)
final += new_header + "\n"

# Data rows
for line in data_lines:
    cols = line.split(",")
    new_cols = [
        cols[i]
        for i, name in feature_names.items()
        if name not in feature_blacklist
    ]
    final += ",".join(new_cols) + "\n"

with open(filename[:-4]+"_CLEAN.csv", "w") as file:
    file.write(final)
