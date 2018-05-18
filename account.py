from configparser import ConfigParser
from getpass import getpass
from codecs import decode, encode

def _get_account_name_from_input():
    return raw_input("Account> ").strip()

def _get_password_from_input():
    return getpass("Password> ").strip()

def _encode_password(pass_str):
    return encode(pass_str.encode(), 'base64').decode()

def _decode_password(pass_str):
    return decode(pass_str.encode(), 'base64').decode()
    
def get_account_from_ini(ini_filename):

    account = ConfigParser(interpolation=None)

    if not (account.read(ini_filename) and account.has_section('Account')):
        user_account = _get_account_name_from_input()
        user_password = _get_password_from_input()
        password_saved = raw_input("Save password?[Y/N]> ").strip().lower() == 'y'

        account.add_section('Account')
        account.set('Account', 'user_account', user_account)
        account.set(
            'Account',
            'user_password',
            _encode_password(user_password)
            if password_saved is True
            else ''
        )
        account.set('Account', 'password_saved', str(password_saved))

        with open(ini_filename, 'w') as ini_file:
            account.write(ini_file)

        return {'user_account': user_account,
                'user_password': user_password}
    return {
        'user_account':
            account.get('Account', 'user_account'),
        'user_password':
            _decode_password(account.get('Account', 'user_password'))
            if account.getboolean('Account', 'password_saved') is True
            else _get_password_from_input()
    }