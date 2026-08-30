from __future__ import annotations

from .core import _load_protocol, _parse_args
from .predict import _predict
from .score import _failure_receipt, _score, _seal
from .source import _authorize, _inventory, _source


def main() -> int:
    args = _parse_args()
    try:
        protocol = _load_protocol(args.protocol.resolve())
        if args.command == "inventory":
            return _inventory(args, protocol)
        if args.command == "source":
            return _source(args, protocol)
        if args.command == "authorize":
            return _authorize(args, protocol)
        if args.command == "predict":
            return _predict(args, protocol)
        if args.command == "seal":
            return _seal(args, protocol)
        if args.command == "score":
            return _score(args, protocol)
        raise AssertionError("unreachable command")
    except BaseException as error:
        _failure_receipt(args, error)
        raise


if __name__ == "__main__":
    raise SystemExit(main())
