package com.example.concurrency;

import java.util.concurrent.ArrayBlockingQueue;
import java.util.concurrent.BlockingQueue;
import java.util.concurrent.ThreadLocalRandom;
import java.util.concurrent.TimeUnit;

/**
 * Pattern 2: Classic producer-consumer using a BlockingQueue.
 *
 * BlockingQueue.put() blocks the producer when the queue is full, and
 * BlockingQueue.take() blocks the consumer when the queue is empty - so we
 * get back-pressure and synchronization for free, without any wait/notify.
 *
 * To signal end-of-stream we use a "poison pill" sentinel. The consumer
 * stops as soon as it pulls one off the queue.
 */
public final class ProducerConsumer {

    private static final int CAPACITY = 5;
    private static final int ITEM_COUNT = 20;
    private static final Integer POISON_PILL = Integer.MIN_VALUE;

    public static void main(String[] args) throws InterruptedException {
        BlockingQueue<Integer> queue = new ArrayBlockingQueue<>(CAPACITY);

        Thread producer = new Thread(new Producer(queue, ITEM_COUNT), "producer");
        Thread consumer = new Thread(new Consumer(queue), "consumer");

        long t0 = System.currentTimeMillis();
        producer.start();
        consumer.start();

        producer.join();
        consumer.join();
        System.out.printf("Done in %d ms%n", System.currentTimeMillis() - t0);
    }

    private record Producer(BlockingQueue<Integer> queue, int count) implements Runnable {
        @Override
        public void run() {
            try {
                for (int i = 1; i <= count; i++) {
                    int item = ThreadLocalRandom.current().nextInt(1, 1000);
                    queue.put(item);
                    System.out.printf("[producer] put %4d   (queue size=%d)%n", item, queue.size());
                    Thread.sleep(50);
                }
                queue.put(POISON_PILL);
            } catch (InterruptedException e) {
                Thread.currentThread().interrupt();
            }
        }
    }

    private record Consumer(BlockingQueue<Integer> queue) implements Runnable {
        @Override
        public void run() {
            try {
                while (true) {
                    Integer item = queue.take();
                    if (item.equals(POISON_PILL)) {
                        System.out.println("[consumer] poison pill - exiting");
                        return;
                    }
                    System.out.printf("[consumer] got %4d  (queue size=%d)%n", item, queue.size());
                    TimeUnit.MILLISECONDS.sleep(120);
                }
            } catch (InterruptedException e) {
                Thread.currentThread().interrupt();
            }
        }
    }
}
