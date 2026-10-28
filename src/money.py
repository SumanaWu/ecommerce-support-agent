"""Exact local amounts; legacy numeric `amount` is presentation compatibility only."""
from decimal import Decimal, InvalidOperation, localcontext


CURRENCY_DIGITS = {'CNY': 2, 'USD': 2}
SAMPLE_CURRENCY = 'CNY'


def major_to_minor(value, currency):
    if currency not in CURRENCY_DIGITS or isinstance(value, bool):
        raise ValueError('金额币种仅支持明确配置的 CNY 或 USD。')
    try:
        amount = Decimal(str(value))
        with localcontext() as context:
            context.prec = 40
            if not amount.is_finite() or amount < 0 or amount > Decimal(2**63 - 1).scaleb(-CURRENCY_DIGITS[currency]):
                raise ValueError()
            if amount != amount.quantize(Decimal(1).scaleb(-CURRENCY_DIGITS[currency])):
                raise ValueError()
            return int(amount.scaleb(CURRENCY_DIGITS[currency]))
    except (InvalidOperation, ValueError, OverflowError):
        raise ValueError('金额必须为非负、有限且可精确表示为整数分的值，不能静默舍入。') from None


def format_amount(amount_minor, currency):
    if currency not in CURRENCY_DIGITS or type(amount_minor) is not int or not 0 <= amount_minor <= 2**63 - 1:
        raise ValueError('无效的整数金额或币种。')
    with localcontext() as context:
        context.prec = 40
        return f"{Decimal(amount_minor).scaleb(-CURRENCY_DIGITS[currency]):.{CURRENCY_DIGITS[currency]}f}"


def money_view(row):
    display = format_amount(row.amount_minor, row.currency)
    return {'amount_minor': row.amount_minor, 'currency': row.currency, 'amount_display': display,
            'amount': float(display)}  # Existing clients may display this; no business calculations use it.
