from typing import List
import re
from collections import defaultdict
from pathlib import Path
import numpy as np
import io
import requests
from PIL import Image
import PIL
import matplotlib.pyplot as plt

class Data():
    def __init__(self, row: List[str]=None):
        """
            self.row (List[str]): 

        Args:
            row (List): Each row of data is shape: (Category, 
                primary_image_url, 
                all_image_urls, 
                attributes, 
                index)
        """
        
        self.row: List[str] = row
        self.category: int = row[0]
        self.primary_image_url: str = row[1]
        self.all_image_urls: List[str] = row[2].split(";")
        self.attributes: defaultdict = self.parse_attributes(row[3])
        self.id = row[4]

    def strip_string(self, s: str):
        # convert to lowercase
        ans = s.lower()
        # Strip any "(" in key
        ans = ans.replace("(", "")
        # Strip any ")" in key
        ans = ans.replace(")", "")
        # Strip space left and right
        ans = ans.strip()
        return ans

    def parse_attributes(self, attr: str) -> defaultdict:
        """ The parsing logic is

        Args:
            attr (str): [description]

        Returns:
            Dict: [description]
        """
        parse_string = attr[1:-1]
        colon_separated = re.split(r"\s*:+\s*", parse_string)
        # Actual return data
        attr_data = defaultdict(list)
        # Current key is guaranteed to be not None
        cur_key = None

        for i, element in enumerate(colon_separated):
            if i == 0:
                cur_key = self.strip_string(element)
                continue

            # Separated by comma
            comma_separated = re.split(r"\s*,\s*", element)
            # If it is not the last item, and parse according to the pattern
            if i != len(colon_separated) - 1:
                for j, element_attr in enumerate(comma_separated):
                    # If it is the last element
                    if j == len(comma_separated) - 1:
                        cur_key = self.strip_string(element_attr)
                    else:
                        attr_data[cur_key].append(self.strip_string(element_attr))
            # Else, all element in the comma separated 
            #   list is the attribute of the last key
            else:
                for j, element_attr in enumerate(comma_separated):
                    attr_data[cur_key].append(self.strip_string(element_attr))
        return attr_data

    def get_image(self, url) -> PIL.Image:
        res = requests.get(url)
        image_bytes = io.BytesIO(res.content)
        img = Image.open(image_bytes)
        # The numpy array of the image
        # img_mat = np.asarray(img)
        return img

    def get_all_images(self) -> List[np.array]:
        """ Return the list crawled images this data object

        Returns:
            List[np.array]: List of np.array (shape= (dim, height, width))
        """
        imgs: List[np.array] = []
        imgs.append(self.get_image(self.primary_image_url))
        
        for url in self.all_image_urls:
            imgs.append(self.get_image(url))
        return imgs

    def show_all_images(self, figure_height: int=10, figure_width: int=10):
        """TODO: NOT DONE! TO BE IMPLEMENTED

        Args:
            figure_height (int, optional): [description]. Defaults to 10.
            figure_width (int, optional): [description]. Defaults to 10.
        """
        imgs = self.get_all_images()
        fig=plt.figure(figsize=(figure_height, figure_width))
        for i, img in enumerate(imgs):
            # fig.add_subplot(i,2,1)
            plt.imshow(img)
        plt.show()
        
    def __str__(self):
        string = ""
        string += "Category: " + str(self.category) + "; "
        string += "Id: " + str(self.id) + "; "
        string += "Attr: " + str(self.attributes)
        return string

############################    Test     ####################################



def test_show_all_images():
    # Preq
    line = "2\thttps://i.ebayimg.com/00/s/MTIwMFgxNjAw/z/qocAAOSw8GJcr4Db/$_12.JPG?set_id=880000500F\thttps://i.ebayimg.com/00/s/MTIwMFgxNjAw/z/qocAAOSw8GJcr4Db/$_12.JPG?set_id=880000500F;https://i.ebayimg.com/00/s/MTIwMFgxNjAw/z/M4QAAOSwJxRcr4DP/$_12.JPG?set_id=880000500F;https://i.ebayimg.com/00/s/MTIwMFgxNjAw/z/TpoAAOSwpLxcr4DY/$_12.JPG?set_id=880000500F;https://i.ebayimg.com/00/s/MTYwMFgxMjAw/z/qQYAAOSw27dcr4DR/$_12.JPG?set_id=880000500F;https://i.ebayimg.com/00/s/MTIwMFgxNjAw/z/wpEAAOSwQXdcr4DZ/$_12.JPG?set_id=880000500F;https://i.ebayimg.com/00/s/MTEyOFgxNjAw/z/OWwAAOSwWdFcr4Dc/$_12.JPG?set_id=880000500F;https://i.ebayimg.com/00/s/MTIwMFgxNjAw/z/q0YAAOSw2S9cr4DN/$_12.JPG?set_id=880000500F;https://i.ebayimg.com/00/s/MTYwMFgxMjAw/z/XX8AAOSwGNFcr4DT/$_12.JPG?set_id=880000500F;https://i.ebayimg.com/00/s/MTYwMFgxMjAw/z/CaQAAOSwdZJcr4DW/$_12.JPG?set_id=880000500F;https://i.ebayimg.com/00/s/NTk3WDgzNg==/z/GEYAAOSwiuRcr4L8/$_12.JPG?set_id=880000500F;https://i.ebayimg.com/00/s/MTIwMFgxNjAw/z/IfAAAOSwvSNcr4L9/$_12.JPG?set_id=880000500F\t(Style:Skateboarding,US Shoe Size (Men's):8.5,Closure:Slip On,Color:White,Country/Region of Manufacture:Vietnam,Euro Size:EUR 41,MPN:VN0A38F7U59,Brand:VANS,Upper Material:Canvas,US Shoe Size (Women's):10,Model:VN0A38F7U59,Product Line:VANS Classic Slip-On)\t7437\n"
    row = line.split("\t")
    try:
        # replace \n for every row of data, in the id
        row[4] = row[4].replace("\n", "")
        dta = Data(row)
    except:
        print(f"cannot add data: {line}; row length: {len(row)}")
    import matplotlib.pyplot as plt
    n_rows = 4
    n_cols = 4
    imgs = dta.get_all_images()
    fig=plt.figure(figsize=(4, 4))
    for i, img in enumerate(imgs):
        # fig.add_subplot(i,2,1)
        # plt.imshow(img)
        img.show()
    # plt.show()

if __name__ == "__main__":
    test_show_all_images()