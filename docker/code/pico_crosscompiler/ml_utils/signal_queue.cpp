#include <cstdint>
#include <list>
#include "signal_queue.h"


SignalQueue::SignalQueue(
    uint32_t maxLength,
    uint32_t channels
) : maxLength(maxLength),
    channels(channels), 
    overflowCounter(0),
    notifyAtEveryNthOverflow(-1),
    eventHandler(nullptr) { }


void SignalQueue::add(int16_t values[]) {
    if (this->isFilled()) {
        this->removeHead();
        if (this->hasEventHandler()) this->overflowCounter++;
    }

    for (uint32_t i = 0; i < this->channels; i++) {
        this->rawQueue.push_back(values[i]);
    }
    
    if (this->isFilled() && this->hasEventHandler()) {
        if (this->overflowNotificationRequired()) {
            this->eventHandler(*this);
        }
        this->overflowCounter %= this->notifyAtEveryNthOverflow;
    }
}

void SignalQueue::copyToBuffer(float buffer[]) {
    std::list<int16_t>::iterator iter = this->rawQueue.begin();

    for (uint32_t i = 0; i < this->rawQueue.size(); i++) {
        buffer[i] = *iter;
        iter++;
    }
}

void SignalQueue::notifyOnOverflowingElement(
        uint32_t atEveryNthElement,
        void (*eventHandler)(SignalQueue&)) {
    this->eventHandler = eventHandler;
    this->notifyAtEveryNthOverflow =  atEveryNthElement;
}


bool SignalQueue::hasEventHandler() {
    return (this->eventHandler != nullptr) && (this->notifyAtEveryNthOverflow != -1);
}


bool SignalQueue::overflowNotificationRequired() {
    return this->overflowCounter == this->notifyAtEveryNthOverflow;
}


bool SignalQueue::isFilled() {
    return this->rawQueue.size() >= this->maxLength;
}


void SignalQueue::removeHead() {
    for (uint32_t i = 0; i < this->channels; i++) {
        this->rawQueue.pop_front();
    }
}
