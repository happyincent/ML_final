# ML final project

## Enviroments

* OS: Ubuntu Gnome 16.04
* CPU: Intel core i5-3210M
* GPU: None

## Dependencies

* Python 2.7

* Keras

* Pillow ( image preprocessing )
  - pip install Pillow
  
* BeautifulSoup ( handle html get from requests )
  - pip install beautifulsoup4
  
* requests ( GET POST requests in test.py )
  - pip install requests
  
* seleniumrequests ( GET POST requests in final.py )
  - pip install selenium-requests
  - Extends Selenium WebDriver classes to include the request function from the Requests library.
  - To use Chrome, should download chromedriver from:
  	https://sites.google.com/a/chromium.org/chromedriver/downloads

## Usage

test.py
```
mkdir ./code_auto
python test.py [model filename]
(set AUTO_SAVE_IMG to True will save both images and label_auto.txt.)
```

final.py ( open browser )
```
python final.py [model filename]
```

my_cnn.py my_nn.py
```
set VCODE_PATH to the path you save the code image.
set LABEL_PATH to the path you save the label.txt.
```

## Training Method & Result

* Check & Label 1500 digital number images in 375 vcodes by human eye

* Pre-train a NN model by the data (1500)

* Test 1000 times, the real accuracy: 84.30%

* Test 25000 times, the real accuracy: 83.58%
  * Save 4*20895 = 83580 digital number images and correct labels

* Train two models (NN, CNN) by the data (83580)

* Test 1000 times, the real accuracy: 96.40% (NN), 99.10% (CNN)

* 2017-06-14 https://www.youtube.com/watch?v=_DXRHwO__l8
