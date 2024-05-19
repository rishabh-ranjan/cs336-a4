import re

# source: https://stackoverflow.com/questions/201323/how-can-i-validate-an-email-address-using-a-regular-expression?page=1&tab=scoredesc#tab-top
email_regex = r"""(?:[a-z0-9!#$%&'*+/=?^_`{|}~-]+(?:\.[a-z0-9!#$%&'*+/=?^_`{|}~-]+)*|"(?:[\x01-\x08\x0b\x0c\x0e-\x1f\x21\x23-\x5b\x5d-\x7f]|\\[\x01-\x09\x0b\x0c\x0e-\x7f])*")@(?:(?:[a-z0-9](?:[a-z0-9-]*[a-z0-9])?\.)+[a-z0-9](?:[a-z0-9-]*[a-z0-9])?|\[(?:(?:(2(5[0-5]|[0-4][0-9])|1[0-9][0-9]|[1-9]?[0-9]))\.){3}(?:(2(5[0-5]|[0-4][0-9])|1[0-9][0-9]|[1-9]?[0-9])|[a-z0-9-]*[a-z0-9]:(?:[\x01-\x08\x0b\x0c\x0e-\x1f\x21-\x5a\x53-\x7f]|\\[\x01-\x09\x0b\x0c\x0e-\x7f])+)\])"""

phone_number_regex = r"""(?:\+?\d{1,3}[-.\s()]*)?(?:\(?\d{1,4}\)?[-.\s]*)?\d{1,4}[-.\s]*\d{1,4}[-.\s]*\d{1,9}"""


email_regex_compiled = re.compile(email_regex)
phone_number_regex_compiled = re.compile(phone_number_regex)


def mask_emails(text: str) -> tuple[str, int]:
    masked_text = email_regex_compiled.sub("|||EMAIL_ADDRESS|||", text)
    num_emails = len(email_regex_compiled.findall(text))
    return masked_text, num_emails


def mask_phone_numbers(text: str) -> tuple[str, int]:
    masked_text = phone_number_regex_compiled.sub("|||PHONE_NUMBER|||", text)
    num_phone_numbers = len(phone_number_regex_compiled.findall(text))
    return masked_text, num_phone_numbers
