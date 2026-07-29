# V14 Source Finalizer

The V14 source finalizer is locked before any real source admission
disposition exists. It consumes only:

- the two preserved pre-lock technical staging failures;
- hash-only carrier and preflight admission dispositions;
- the immutable staging queue;
- the frozen exclusion and synthetic-control artifacts.

It requires a contiguous queue prefix and stops exactly when the twelfth
admitted source is reached. It writes an immutable selection ledger and the
three-fold source lock together. A queue gap, a thirteenth admission, an
unmatched hash, an unaccepted preflight, or a changed implementation aborts
without producing the final directory.

The finalizer does not read object response, identity targets, metrics, source
outcomes, target artifacts, or held-v8 artifacts and processes.
