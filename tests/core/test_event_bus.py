import pytest
from ia_test.core.event_bus import EventBus
from ia_test.core.events import KnowledgeAcquired, QuestionGenerated


async def test_subscriber_receives_published_event():
    bus = EventBus()
    received = []

    async def handler(event):
        received.append(event)

    bus.subscribe(KnowledgeAcquired, handler)
    event = KnowledgeAcquired(entry_id="abc", entry_type="knowledge", confidence=0.8)
    await bus.publish(event)

    assert len(received) == 1
    assert received[0].entry_id == "abc"
    assert received[0].confidence == 0.8


async def test_unrelated_subscriber_does_not_receive_event():
    bus = EventBus()
    received = []

    async def handler(event):
        received.append(event)

    bus.subscribe(QuestionGenerated, handler)
    await bus.publish(KnowledgeAcquired(entry_id="abc"))

    assert len(received) == 0


async def test_multiple_subscribers_all_receive():
    bus = EventBus()
    received_a = []
    received_b = []

    async def handler_a(e):
        received_a.append(e)

    async def handler_b(e):
        received_b.append(e)

    bus.subscribe(KnowledgeAcquired, handler_a)
    bus.subscribe(KnowledgeAcquired, handler_b)

    await bus.publish(KnowledgeAcquired(entry_id="xyz"))

    assert len(received_a) == 1
    assert len(received_b) == 1


async def test_failing_handler_does_not_block_others():
    bus = EventBus()
    received = []

    async def bad_handler(e):
        raise RuntimeError("boom")

    async def good_handler(e):
        received.append(e)

    bus.subscribe(KnowledgeAcquired, bad_handler)
    bus.subscribe(KnowledgeAcquired, good_handler)

    await bus.publish(KnowledgeAcquired(entry_id="test"))

    assert len(received) == 1


async def test_no_subscribers_publish_is_noop():
    bus = EventBus()
    await bus.publish(KnowledgeAcquired(entry_id="orphan"))
