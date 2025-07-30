import os
import re
import base64
from typing import List, Tuple, Optional, Dict
import fitz
import shapely.geometry as sg
from shapely.geometry.base import BaseGeometry
from shapely.validation import explain_validity
import concurrent.futures
import logging
from openai import OpenAI

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# This Default Prompt Using Chinese and could be changed to other languages.

DEFAULT_PROMPT = """Using markdown syntax, convert the recognized text in the image to markdown format output. You must:
1. Output and use the same language detected in the image, for example, if a field is detected as English, the output must be in English.
2. Do not explain or output irrelevant text, directly output the content in the image. For example, it is strictly forbidden to output examples like "The following is the markdown text I generated based on the image content:"; instead, output the markdown text directly.
3. Do not wrap the content with ```markdown ```, use $$ $$ for block formulas, $ $ for inline formulas, ignore long straight lines, and ignore page numbers.
To reiterate, do not explain or output irrelevant text, just output the content in the image directly.
"""
DEFAULT_RECT_PROMPT = """Some areas are marked in the image with a red box and label (%s). If the area is a table or image, use ![]() to insert it into the output; otherwise, output the text content directly.
"""
DEFAULT_ROLE_PROMPT = """You are a PDF document parser. Output the content of the image using markdown and LaTeX syntax.
"""

def _is_near(rect1, rect2, distance = 20):
    """
    Check if two rectangles are close, i.e., if their distance is less than the target distance.
    @param rect1: Rectangle 1
    @param rect2: Rectangle 2
    @param distance: Target distance
    @return: Whether they are close
    """
    return rect1.buffer(0.1).distance(rect2.buffer(0.1)) < distance

def _is_horizontal_near(rect1, rect2, distance = 100):
    """
    Check if two rectangles are horizontally close, especially if one is a horizontal line.
    @param rect1: Rectangle 1
    @param rect2: Rectangle 2
    @param distance: Target distance
    @return: Whether they are horizontally close
    """
    result = False
    if abs(rect1.bounds[3] - rect1.bounds[1]) < 0.1 or abs(rect2.bounds[3] - rect2.bounds[1]) < 0.1:
        if abs(rect1.bounds[0] - rect2.bounds[0]) < 0.1 and abs(rect1.bounds[2] - rect2.bounds[2]) < 0.1:
            result = abs(rect1.bounds[3] - rect2.bounds[3]) < distance
    return result

def _union_rects(rect1, rect2):
    """
    Merge two rectangles.
    @param rect1: Rectangle 1
    @param rect2: Rectangle 2
    @return: Merged rectangle
    """
    return sg.box(*(rect1.union(rect2).bounds))

def _merge_rects(rect_list, distance = 20, horizontal_distance = None):
    """
    Merge rectangles in the list if they are closer than the target distance.
    @param rect_list: List of rectangles
    @param distance: Target distance
    @param horizontal_distance: Horizontal target distance
    @return: Merged rectangle list
    """
    merged = True
    while merged:
        merged = False
        new_rect_list = []
        while rect_list:
            rect = rect_list.pop(0)
            for other_rect in rect_list:
                if _is_near(rect, other_rect, distance) or (
                        horizontal_distance and _is_horizontal_near(rect, other_rect, horizontal_distance)):
                    rect = _union_rects(rect, other_rect)
                    rect_list.remove(other_rect)
                    merged = True
            new_rect_list.append(rect)
        rect_list = new_rect_list
    return rect_list

def _adsorb_rects_to_rects(source_rects, target_rects, distance=10):
    """
    Attach a set of rectangles to another set if they are closer than the target distance.
    @param source_rects: Source rectangle list
    @param target_rects: Target rectangle list
    @param distance: Target distance
    @return: Updated source and target rectangle lists after attachment
    """
    new_source_rects = []
    for text_area_rect in source_rects:
        adsorbed = False
        for index, rect in enumerate(target_rects):
            if _is_near(text_area_rect, rect, distance):
                rect = _union_rects(text_area_rect, rect)
                target_rects[index] = rect
                adsorbed = True
                break
        if not adsorbed:
            new_source_rects.append(text_area_rect)
    return new_source_rects, target_rects

def _parse_rects(page):
    """
    Parse drawings in the page and merge adjacent rectangles.
    @param page: Page
    @return: List of rectangles
    """

    # Extract drawing content
    drawings = page.get_drawings()

    # Ignore horizontal lines shorter than 30 units
    is_short_line = lambda x: abs(x['rect'][3] - x['rect'][1]) < 1 and abs(x['rect'][2] - x['rect'][0]) < 30
    drawings = [drawing for drawing in drawings if not is_short_line(drawing)]

    # Convert to shapely rectangles
    rect_list = [sg.box(*drawing['rect']) for drawing in drawings]

    # Extract image areas
    images = page.get_image_info()
    image_rects = [sg.box(*image['bbox']) for image in images]

    # Merge drawings and images
    rect_list += image_rects

    merged_rects = _merge_rects(rect_list, distance=10, horizontal_distance=100)
    merged_rects = [rect for rect in merged_rects if explain_validity(rect) == 'Valid Geometry']

    # Separate large and small text areas: merge large text closely, merge small text if near
    is_large_content = lambda x: (len(x[4]) / max(1, len(x[4].split('\n')))) > 5
    small_text_area_rects = [sg.box(*x[:4]) for x in page.get_text('blocks') if not is_large_content(x)]
    large_text_area_rects = [sg.box(*x[:4]) for x in page.get_text('blocks') if is_large_content(x)]
    _, merged_rects = _adsorb_rects_to_rects(large_text_area_rects, merged_rects, distance=0.1) # Fully overlapping
    _, merged_rects = _adsorb_rects_to_rects(small_text_area_rects, merged_rects, distance=5) # Nearby

    # Merge again
    merged_rects = _merge_rects(merged_rects, distance=10)

    # Filter out rectangles that are too small
    merged_rects = [rect for rect in merged_rects if rect.bounds[2] - rect.bounds[0] > 20 and rect.bounds[3] - rect.bounds[1] > 20]

    return [rect.bounds for rect in merged_rects]

def _parse_pdf_to_images(pdf_path, output_dir = './'):
    """
    Parse PDF file to images and save to the output directory.
    @param pdf_path: PDF file path
    @param output_dir: Output directory
    @return: List of image info (image path, list of rectangle image paths)
    """
    # Open PDF file
    pdf_document = fitz.open(pdf_path)
    image_infos = []

    for page_index, page in enumerate(pdf_document):
        logging.info(f'parse page: {page_index}')
        rect_images = []
        rects = _parse_rects(page)
        for index, rect in enumerate(rects):
            fitz_rect = fitz.Rect(rect)
            # Save page as image
            pix = page.get_pixmap(clip=fitz_rect, matrix=fitz.Matrix(4, 4))
            name = f'{page_index}_{index}.png'
            pix.save(os.path.join(output_dir, name))
            rect_images.append(name)
            # Draw red rectangle on the page
            big_fitz_rect = fitz.Rect(fitz_rect.x0 - 1, fitz_rect.y0 - 1, fitz_rect.x1 + 1, fitz_rect.y1 + 1)
            # Hollow rectangle
            page.draw_rect(big_fitz_rect, color=(1, 0, 0), width=1)
            # Draw filled rectangle (commented out)
            # page.draw_rect(big_fitz_rect, color=(1, 0, 0), fill=(1, 0, 0))
            # Write the rectangle's index name at the top left inside the rectangle, with some offset
            text_x = fitz_rect.x0 + 2
            text_y = fitz_rect.y0 + 10
            text_rect = fitz.Rect(text_x, text_y - 9, text_x + 80, text_y + 2)
            # Draw white background rectangle
            page.draw_rect(text_rect, color=(1, 1, 1), fill=(1, 1, 1))
            # Insert text with white background
            page.insert_text((text_x, text_y), name, fontsize=10, color=(1, 0, 0))
        page_image_with_rects = page.get_pixmap(matrix=fitz.Matrix(3, 3))
        page_image = os.path.join(output_dir, f'{page_index}.png')
        page_image_with_rects.save(page_image)
        image_infos.append((page_image, rect_images))

    pdf_document.close()
    return image_infos

def _remove_markdown_backticks(content: str) -> str:
    """
    Remove ``` strings from markdown content.
    """
    if '```markdown' in content:
        content = content.replace('```markdown\n', '')
        last_backticks_pos = content.rfind('```')
        if last_backticks_pos != -1:
            content = content[:last_backticks_pos] + content[last_backticks_pos + 3:]
    return content

def parse_pdf(
        pdf_path: str,
        output_dir: str = './',
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        model: str = 'gpt-4o',
        gpt_worker: int = 1,
        prompt = DEFAULT_PROMPT,
        rect_prompt = DEFAULT_RECT_PROMPT,
        role_prompt = DEFAULT_ROLE_PROMPT,
) -> Tuple[str, List[str]]:
    """
    Parse PDF file to a markdown file.
    @param pdf_path: PDF file path
    @param output_dir: Output directory
    @return: Parsed markdown content, list of rectangle image paths
    """
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    image_infos = _parse_pdf_to_images(pdf_path, output_dir=output_dir)
    
    # Process images with GPT
    def _process_page(index: int, image_info: Tuple[str, List[str]]) -> Tuple[int, str]:
        # Use OpenAI client instead of Agent
        client = OpenAI(api_key=api_key, base_url=base_url)
        page_image, rect_images = image_info
        local_prompt = prompt
        if rect_images:
            local_prompt += rect_prompt + ', '.join(rect_images)
        
        # Open image file
        with open(page_image, "rb") as image_file:
            # Call OpenAI API
            try:
                response = client.chat.completions.create(
                    model=model,
                    messages=[
                        {"role": "system", "content": role_prompt},
                        {"role": "user", "content": [
                            {"type": "text", "text": local_prompt},
                            {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{base64.b64encode(image_file.read()).decode('utf-8')}"}}
                        ]}
                    ]
                )

                # Check if response.choices is None
                if not response.choices:
                    print(response)
                    return index, f"Error: Empty choices in API response for page {index+1}"
                    
                content = response.choices[0].message.content
                return index, content
            except Exception as e:
                # Catch all exceptions and return error message
                return index, f"Error processing page {index+1}: {str(e)}"

    contents = [None] * len(image_infos)
    with concurrent.futures.ThreadPoolExecutor(max_workers=gpt_worker) as executor:
        futures = [executor.submit(_process_page, index, image_info) for index, image_info in enumerate(image_infos)]
        for future in concurrent.futures.as_completed(futures):
            index, content = future.result()
            content = _remove_markdown_backticks(content)
            contents[index] = content

    # Save parsed markdown file
    output_path = os.path.join(output_dir, 'output.md')
    content = '\n\n'.join(contents)
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(content)

    # Delete intermediate images
    all_rect_images = []
    for page_image, rect_images in image_infos:
        if os.path.exists(page_image):
            os.remove(page_image)
        all_rect_images.extend(rect_images)

    return content, all_rect_images
