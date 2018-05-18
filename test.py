from predict import *
from account import *

import requests
from bs4 import BeautifulSoup
from StringIO import StringIO
from PIL import Image, ImageOps, ImageFilter
import numpy as np
import sys
from keras.models import load_model

TEST_NUM = 10
SHOW_IMG = False # True if TEST_NUM is small!
AUTO_SAVE_IMG = False
MODEL_FILENAME = 'model_cnn.h5'
INI_FILENAME = 'account.ini'
DRIVER_PATH = './chromedriver'

LOGIN_URL = 'http://selcrs.nsysu.edu.tw/'
CHECK_URL = 'http://selcrs.nsysu.edu.tw/menu4/Studcheck.asp'
WRONG_MSG = "Wrong Validation Code"


def get_vcode(session):
    start_res = session.get(LOGIN_URL)

    soup = BeautifulSoup(start_res.text, 'html.parser')
    VCODE_URL = LOGIN_URL + soup.find(id='imgVC')['src']

    vcode_res = session.get(VCODE_URL)
    img = Image.open(StringIO(vcode_res.content))

    return img

def convert_vcode(img):
    img = ImageOps.invert(img)
    img = img.filter(ImageFilter.ModeFilter)
    img = img.convert('L')

    vcode = []

    for j in range(0, 4):
        quater = img.crop((img.width/4*j, 0,
                          img.width/4*(j+1), img.height))
        (width, height) = quater.size
        quater = np.asarray(quater, dtype="uint8")
        quater.flags.writeable = True

        threshold = np.unique(quater)[0]
        quater[quater > threshold] = 255
        quater[quater == threshold] = 0

        vcode.append(quater)

    vcode = np.asarray(vcode)
    return vcode

def login(session, account, vcode_str):

    params = {
        'stuid': account['user_account'],
        'SPassword': account['user_password'],
        'ValidCode': vcode_str
    }

    login_res = session.post(CHECK_URL, data=params)
    login_res.encoding = 'big5'

    soup = BeautifulSoup(login_res.text, 'html.parser')

    return not(
        soup.find(string=WRONG_MSG) == WRONG_MSG
    )

def main():
    model_name = MODEL_FILENAME
    if len(sys.argv) == 2:
        model_name = sys.argv[1]
    print('Use model: ' + model_name)
    model = load_model(model_name)

    account_info = get_account_from_ini(INI_FILENAME)
    print('ID: "' + account_info['user_account'] + '" start login test ...')

    count = 0
    fout = open('label_auto.txt', 'w')

    for i in range(1, TEST_NUM+1):
        session = requests.Session()

        img = get_vcode(session)
        vcode = convert_vcode(img)

        print('Start predict (' + str(i) + ')!')
        vcode_str = predict(vcode, model_name, model)
        print('Get vcode: "' + vcode_str + '"')

        if_login = login(session, account_info, vcode_str)            
        print('Result: ' + str(if_login))

        if not if_login and SHOW_IMG:
            ImageOps.invert(img)\
            .filter(ImageFilter.ModeFilter)\
            .convert('L')\
            .show()

        if if_login:
            count = count + 1

            if AUTO_SAVE_IMG:
                print('Success verified: ' + str(count))
                vcode *= 255
                for j in range(0, 4):
                    fout.write(vcode_str[j]+"\n")
                    Image.fromarray(vcode[j])\
                    .save('code_auto/code' \
                    + str((count-1)*4+j+1) + '.bmp')


        session.close()
        
    fout.close()
    print( 'Accuracy: ' + str(format(count / float(TEST_NUM) * 100.0, '.2f')) + ' %' )
    
    
if __name__ == '__main__':
    main()