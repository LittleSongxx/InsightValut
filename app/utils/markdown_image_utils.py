from typing import List


def extract_markdown_image_urls(text: str) -> List[str]:
    if not text:
        return []

    urls: List[str] = []
    seen = set()
    cursor = 0
    text_length = len(text)

    while cursor < text_length:
        image_start = text.find("![", cursor)
        if image_start == -1:
            break

        alt_close = text.find("](", image_start + 2)
        if alt_close == -1:
            cursor = image_start + 2
            continue

        url_start = alt_close + 2
        depth = 1
        index = url_start

        while index < text_length:
            char = text[index]
            if char == "\\":
                index += 2
                continue
            if char == "(":
                depth += 1
            elif char == ")":
                depth -= 1
                if depth == 0:
                    break
            index += 1

        if depth != 0:
            cursor = alt_close + 2
            continue

        raw_url = text[url_start:index].strip()
        if raw_url.startswith("<") and raw_url.endswith(">"):
            raw_url = raw_url[1:-1].strip()

        if raw_url and raw_url not in seen:
            seen.add(raw_url)
            urls.append(raw_url)

        cursor = index + 1

    return urls
