#!/usr/bin/python


if __name__ == "__main__":
    with open("results.csv", "r") as f:
        results = f.read().splitlines()

    iframe_idx = results[0].split('"').index("Input.iframe_url")
    iframes = []
    for r in results[1:]:
        iframes.append(r.split('"')[iframe_idx])

    with open("iframes.csv", "r") as f:
        all_iframes = f.read().splitlines()

    rest_iframes = set(all_iframes[1:]) - set(iframes)

    with open("rest_iframes.csv", "w") as f:
        f.write("iframe_url\n")
        for ri in rest_iframes:
            f.write(ri + "\n")
