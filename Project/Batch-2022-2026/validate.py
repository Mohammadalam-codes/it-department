import datetime

def preprocess():
    expiration_date = datetime.date(2026, 6, 30)
    today = datetime.date.today()
    msg=""
    if today > expiration_date:
        msg="invalid"
    else:
        msg="valid"

    return msg
