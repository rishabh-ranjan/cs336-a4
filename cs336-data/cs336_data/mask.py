import re
import phonenumbers
from phonenumbers import PhoneNumberMatcher

# source: https://stackoverflow.com/questions/201323/how-can-i-validate-an-email-address-using-a-regular-expression?page=1&tab=scoredesc#tab-top
email_regex = r"""(?:[a-z0-9!#$%&'*+/=?^_`{|}~-]+(?:\.[a-z0-9!#$%&'*+/=?^_`{|}~-]+)*|"(?:[\x01-\x08\x0b\x0c\x0e-\x1f\x21\x23-\x5b\x5d-\x7f]|\\[\x01-\x09\x0b\x0c\x0e-\x7f])*")@(?:(?:[a-z0-9](?:[a-z0-9-]*[a-z0-9])?\.)+[a-z0-9](?:[a-z0-9-]*[a-z0-9])?|\[(?:(?:(2(5[0-5]|[0-4][0-9])|1[0-9][0-9]|[1-9]?[0-9]))\.){3}(?:(2(5[0-5]|[0-4][0-9])|1[0-9][0-9]|[1-9]?[0-9])|[a-z0-9-]*[a-z0-9]:(?:[\x01-\x08\x0b\x0c\x0e-\x1f\x21-\x5a\x53-\x7f]|\\[\x01-\x09\x0b\x0c\x0e-\x7f])+)\])"""

# source: chatgpt
ip_regex = r"""\b(?:\d{1,3}\.){3}\d{1,3}\b"""

email_regex_compiled = re.compile(email_regex)
ip_regex_compiled = re.compile(ip_regex)


def mask_emails(text: str) -> tuple[str, int]:
    masked_text = email_regex_compiled.sub("|||EMAIL_ADDRESS|||", text)
    num_emails = len(email_regex_compiled.findall(text))
    return masked_text, num_emails


def mask_phone_numbers(text: str) -> tuple[str, int]:
    matches = PhoneNumberMatcher(text, "US", leniency=phonenumbers.Leniency.POSSIBLE)
    count = 0
    for match in matches:
        count += 1
        text = text.replace(match.raw_string, "|||PHONE_NUMBER|||")
    return text, count


def mask_ips(text: str) -> tuple[str, int]:
    masked_text = ip_regex_compiled.sub("|||IP_ADDRESS|||", text)
    num_ips = len(ip_regex_compiled.findall(text))
    return masked_text, num_ips
