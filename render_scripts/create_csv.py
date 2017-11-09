import csv


def create_csv(csv_fname, fn_pairs, prefix):
    with open(csv_fname, "w") as csvfile:
        filewriter = csv.writer(csvfile, delimiter=",", quotechar="|", quoting=csv.QUOTE_MINIMAL)
        filewriter.writerow(["image_url"])
        for i, j in fn_pairs:
            filewriter.writerow(["%s%d/chunky.%d.png" % (prefix, i, j)])


if __name__ == "__main__":
    prefix = "https://dl.fbaipublicfiles.com/minecraft2dvision/"

    fn_pairs = []
    for scene_id in range(1000, 2000):
        for angle_id in [0, 90]:
            fn_pairs.append((scene_id, angle_id))

    create_csv("toy.csv", fn_pairs, prefix)
