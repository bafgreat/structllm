"""Synthetic, tool-grounded training data for StructLLM.

Requests are written by a language model from structured scenario specs; the
assistant side is produced by a teacher model acting through the same bounded
orchestrator the trained model will use, against a development StructSub
server. Cheap operations are executed for real; expensive ones are never sent
and receive a clearly marked mock approval hold. Every record states which of
its tool results were executed and which were mocked.
"""
