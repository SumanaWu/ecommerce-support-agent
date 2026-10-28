"""Executable local rules; retrieved policy prose never configures this module."""
import hashlib
import json


REFUND_RULE_VERSION = 'local-refund-v1'


def evaluate_refund_policy(order, refund):
    status = refund.status if refund else 'none'
    result = {'order_id': order.order_id, 'rule_version': REFUND_RULE_VERSION,
              'checks': {'delivered': order.status == 'delivered', 'no_existing_refund': status == 'none'}}
    if order.status != 'delivered':
        return {**result, 'eligible': False, 'reason_code': 'not_delivered',
                'reason': f"Order status is '{order.status}'; only delivered orders are refund-eligible."}
    if status != 'none':
        return {**result, 'eligible': False, 'reason_code': 'existing_refund',
                'reason': f"A refund is already '{status}' for this order."}
    return {**result, 'eligible': True, 'reason_code': 'eligible',
            'reason': 'Order is delivered and has no existing refund.'}


def business_fingerprint(order, payment, refund):
    """Works on migration row dictionaries as well as current ORM facts."""
    facts = [order['customer_id'], order['version'], order['status'], order['amount_minor'], order['currency'],
             order['item'], str(order['order_date']),
             [payment['payment_id'], payment['status'], payment['method'], payment['amount_minor'], payment['currency']] if payment else None,
             [refund['refund_id'], refund['status'], refund['reason']] if refund else None,
             REFUND_RULE_VERSION]
    return hashlib.sha256(json.dumps(facts, sort_keys=True, ensure_ascii=False, separators=(',', ':')).encode()).hexdigest()
