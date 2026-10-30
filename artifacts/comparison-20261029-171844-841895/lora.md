# RAG + LoRA: raw answers

No scoring or quality assessment has been performed.

Model: `HuggingFaceTB/SmolLM2-135M-Instruct`. Adapter trained for 8 epochs.

Greedy decoding; maximum 80 new tokens. Identical evidence and prompts for both systems.

Answers are preserved as generated, without citation filtering or replacement. When no evidence meets the threshold, both paths return a pipeline fallback without calling the model.

## eval-refund-ticket

A digital workshop starts in 12 hours. Can the customer still receive a refund?

### Shared retrieved evidence

[refunds] paragraph 1, similarity 0.743

Digital workshop tickets are refundable until 24 hours before the workshop starts. Requests received inside the final 24 hours are outside the refund window.

[refunds] paragraph 0, similarity 0.475

Standard purchases are refundable within 30 calendar days when the customer provides the order ID. Approved refunds return to the original payment method.

[refunds] paragraph 2, similarity 0.406

For a duplicate payment, collect the invoice number and both payment references. Finance reviews a complete duplicate-payment request within three business days.

### RAG + LoRA

```text
Yes, the customer can still receive a refund. The digital workshop tickets are refundable until 24 hours before the workshop starts. Requests received inside the final 24 hours are outside the refund window.
```

## eval-duplicate-payment

What must support collect for a duplicate payment investigation?

### Shared retrieved evidence

[refunds] paragraph 2, similarity 0.525

For a duplicate payment, collect the invoice number and both payment references. Finance reviews a complete duplicate-payment request within three business days.

[refunds] paragraph 0, similarity 0.359

Standard purchases are refundable within 30 calendar days when the customer provides the order ID. Approved refunds return to the original payment method.

[refunds] paragraph 1, similarity 0.321

Digital workshop tickets are refundable until 24 hours before the workshop starts. Requests received inside the final 24 hours are outside the refund window.

### RAG + LoRA

```text
Collect the invoice number and both payment references to ensure a complete refund. [refunds]
```

## eval-authenticator

A customer lost their authenticator. What information and process should support use?

### Shared retrieved evidence

[account-security] paragraph 1, similarity 0.607

For account recovery, ask for the registered email address and send a secure recovery link. If a customer lost access to an authenticator, identity verification must use the secure upload portal, never ordinary chat.

[account-security] paragraph 2, similarity 0.398

For a suspicious login, recommend resetting the password, signing out other sessions, and enabling multi-factor authentication.

[response-style] paragraph 0, similarity 0.362

Start with the direct answer. Follow with the next action the customer or support colleague should take.

### RAG + LoRA

```text
Ask for the registered email address and send a secure recovery link. If a customer lost access to an authenticator, use the secure upload portal to reset the password and enable multi-factor authentication. Follow the instructions to ensure the authenticator is properly secured. [account-security]
```

## eval-passport

Should a customer paste a passport scan into chat to recover an account?

### Shared retrieved evidence

[account-security] paragraph 1, similarity 0.606

For account recovery, ask for the registered email address and send a secure recovery link. If a customer lost access to an authenticator, identity verification must use the secure upload portal, never ordinary chat.

[account-security] paragraph 2, similarity 0.363

For a suspicious login, recommend resetting the password, signing out other sessions, and enabling multi-factor authentication.

[incidents] paragraph 1, similarity 0.359

Diagnostic logs may include request IDs and timestamps, but must not include passwords, authentication tokens, one-time passcodes, or full payment-card details.

### RAG + LoRA

```text
Password recovery is not recommended. [account-security]
```

## eval-packed-address

Can support change a delivery address after the warehouse packed the order?

### Shared retrieved evidence

[shipping] paragraph 1, similarity 0.841

A delivery address can be changed only before the warehouse packs the order. After packing, support cannot change the address. After dispatch, the customer may contact the carrier using the tracking number to request an available delivery option.

### RAG + LoRA

```text
Yes, support can change the address before the warehouse packs the order. After packing, support cannot change the address.
```

## eval-standard-shipping

What is the usual standard shipping time?

### Shared retrieved evidence

[shipping] paragraph 0, similarity 0.654

Standard shipping usually takes three to five business days. Express shipping usually takes one to two business days.

### RAG + LoRA

```text
Standard shipping usually takes three to five business days. Express shipping usually takes one to two business days. [shipping]
```

## eval-annual-refund

Can an annual subscription receive a partial refund after 45 days?

### Shared retrieved evidence

[subscriptions] paragraph 1, similarity 0.705

Annual subscriptions are not partially refunded after the 30-day refund window. Support may record cancellation so the plan does not renew.

[refunds] paragraph 0, similarity 0.572

Standard purchases are refundable within 30 calendar days when the customer provides the order ID. Approved refunds return to the original payment method.

[subscriptions] paragraph 0, similarity 0.484

Monthly subscriptions can be cancelled before the next renewal date. Access continues until the end of the paid billing period.

### RAG + LoRA

```text
No, an annual subscription cannot receive a partial refund after the 30-day refund window. The refund is only available within the 30-day window.
```

## eval-pause

How often and for how long can an active subscription be paused?

### Shared retrieved evidence

[subscriptions] paragraph 2, similarity 0.891

An active subscription may be paused once per calendar quarter for up to 30 days. The pause begins on the date selected by the customer.

[subscriptions] paragraph 0, similarity 0.672

Monthly subscriptions can be cancelled before the next renewal date. Access continues until the end of the paid billing period.

[subscriptions] paragraph 1, similarity 0.483

Annual subscriptions are not partially refunded after the 30-day refund window. Support may record cancellation so the plan does not renew.

### RAG + LoRA

```text
The approved context suggests a pause on the date selected by the customer, so the subscription can be cancelled before the next renewal date. [subscriptions]
```

## eval-outage-updates

How frequently should updates be published during a confirmed outage?

### Shared retrieved evidence

[incidents] paragraph 0, similarity 0.708

During a confirmed service outage, publish status updates at least every 30 minutes until recovery. Direct customers to the public status page for the latest confirmed information.

[subscriptions] paragraph 2, similarity 0.367

An active subscription may be paused once per calendar quarter for up to 30 days. The pause begins on the date selected by the customer.

[subscriptions] paragraph 0, similarity 0.317

Monthly subscriptions can be cancelled before the next renewal date. Access continues until the end of the paid billing period.

### RAG + LoRA

```text
Publish status updates at least every 30 minutes until recovery. Direct customers to the public status page for the latest confirmed information.
```

## eval-log-secrets

May diagnostic logs sent to support contain authentication tokens?

### Shared retrieved evidence

[incidents] paragraph 1, similarity 0.654

Diagnostic logs may include request IDs and timestamps, but must not include passwords, authentication tokens, one-time passcodes, or full payment-card details.

[account-security] paragraph 1, similarity 0.344

For account recovery, ask for the registered email address and send a secure recovery link. If a customer lost access to an authenticator, identity verification must use the secure upload portal, never ordinary chat.

[account-security] paragraph 2, similarity 0.320

For a suspicious login, recommend resetting the password, signing out other sessions, and enabling multi-factor authentication.

### RAG + LoRA

```text
Yes, diagnostic logs may include password, authentication tokens, and one-time passcodes. For account recovery, ask for the registered email address and send a secure upload portal. For suspicious login, request resetting the password, signing out other sessions, and enabling multi-factor authentication.
```

## eval-bank-chargeback

How many days will my external bank take to resolve a chargeback for a duplicate payment?

### Shared retrieved evidence

[refunds] paragraph 2, similarity 0.662

For a duplicate payment, collect the invoice number and both payment references. Finance reviews a complete duplicate-payment request within three business days.

[refunds] paragraph 0, similarity 0.605

Standard purchases are refundable within 30 calendar days when the customer provides the order ID. Approved refunds return to the original payment method.

[refunds] paragraph 1, similarity 0.473

Digital workshop tickets are refundable until 24 hours before the workshop starts. Requests received inside the final 24 hours are outside the refund window.

### RAG + LoRA

```text
The standard refund window is three business days. Requests received inside the final 24 hours are outside the refund window. [refunds]
```

## eval-shipping-compensation

What compensation amount is guaranteed if express shipping arrives late?

### Shared retrieved evidence

[shipping] paragraph 0, similarity 0.549

Standard shipping usually takes three to five business days. Express shipping usually takes one to two business days.

[shipping] paragraph 1, similarity 0.378

A delivery address can be changed only before the warehouse packs the order. After packing, support cannot change the address. After dispatch, the customer may contact the carrier using the tracking number to request an available delivery option.

[refunds] paragraph 1, similarity 0.361

Digital workshop tickets are refundable until 24 hours before the workshop starts. Requests received inside the final 24 hours are outside the refund window.

### RAG + LoRA

```text
Express shipping usually arrives three to five business days before the workshop starts. Express shipping usually takes one to two business days before the workshop starts. [shipping]
```

## eval-cafeteria

Where can I find next week's cafeteria menu?

### Shared retrieved evidence

No passage met the threshold. The outputs below are pipeline fallbacks.

### RAG + LoRA

```text
The approved material is insufficient. Escalate for manual review.
```

## eval-bicycle

Can I reserve a bicycle parking space near the office?

### Shared retrieved evidence

No passage met the threshold. The outputs below are pipeline fallbacks.

### RAG + LoRA

```text
The approved material is insufficient. Escalate for manual review.
```
