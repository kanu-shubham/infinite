package com.bookmyshow.config;

import com.bookmyshow.events.BookingCompensateEvent;
import com.bookmyshow.events.PaymentChargedEvent;
import com.bookmyshow.events.PaymentFailedEvent;
import com.bookmyshow.events.SagaTopics;
import com.bookmyshow.events.SeatsReservedEvent;
import org.apache.kafka.clients.admin.NewTopic;
import org.apache.kafka.clients.consumer.ConsumerConfig;
import org.apache.kafka.common.serialization.StringDeserializer;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.context.annotation.Bean;
import org.springframework.context.annotation.Configuration;
import org.springframework.kafka.annotation.EnableKafka;
import org.springframework.kafka.config.ConcurrentKafkaListenerContainerFactory;
import org.springframework.kafka.config.TopicBuilder;
import org.springframework.kafka.core.ConsumerFactory;
import org.springframework.kafka.core.DefaultKafkaConsumerFactory;
import org.springframework.kafka.listener.DefaultErrorHandler;
import org.springframework.kafka.support.serializer.ErrorHandlingDeserializer;
import org.springframework.kafka.support.serializer.JsonDeserializer;
import org.springframework.util.backoff.FixedBackOff;

import java.util.HashMap;
import java.util.Map;

/**
 * Saga Kafka wiring.
 *
 * Why a custom listener factory:
 *   - Saga topics carry several event types. We register a type mapping so the
 *     JsonDeserializer can pick the right class without needing the producer's FQN
 *     (lets us evolve packages independently across services).
 *   - We attach a DefaultErrorHandler with a fixed backoff so a poisoned record
 *     retries 3 times then is logged + skipped (production: route to a DLQ topic).
 */
@EnableKafka
@Configuration
public class SagaKafkaConfig {

    @Value("${spring.kafka.bootstrap-servers}")
    private String bootstrapServers;

    @Bean
    public NewTopic seatsReservedTopic() {
        return TopicBuilder.name(SagaTopics.SEATS_RESERVED).partitions(3).replicas(1).build();
    }

    @Bean
    public NewTopic paymentChargedTopic() {
        return TopicBuilder.name(SagaTopics.PAYMENT_CHARGED).partitions(3).replicas(1).build();
    }

    @Bean
    public NewTopic paymentFailedTopic() {
        return TopicBuilder.name(SagaTopics.PAYMENT_FAILED).partitions(3).replicas(1).build();
    }

    @Bean
    public NewTopic notificationSentTopic() {
        return TopicBuilder.name(SagaTopics.NOTIFICATION_SENT).partitions(3).replicas(1).build();
    }

    @Bean
    public NewTopic compensateTopic() {
        return TopicBuilder.name(SagaTopics.BOOKING_COMPENSATE).partitions(3).replicas(1).build();
    }

    @Bean
    public ConsumerFactory<String, Object> sagaConsumerFactory() {
        Map<String, Object> props = new HashMap<>();
        props.put(ConsumerConfig.BOOTSTRAP_SERVERS_CONFIG, bootstrapServers);
        props.put(ConsumerConfig.AUTO_OFFSET_RESET_CONFIG, "earliest");
        props.put(ConsumerConfig.KEY_DESERIALIZER_CLASS_CONFIG, StringDeserializer.class);
        // Wrap value deserializer so a single bad record doesn't poison the partition.
        props.put(ConsumerConfig.VALUE_DESERIALIZER_CLASS_CONFIG, ErrorHandlingDeserializer.class);
        props.put(ErrorHandlingDeserializer.VALUE_DESERIALIZER_CLASS, JsonDeserializer.class);
        props.put(JsonDeserializer.TRUSTED_PACKAGES, "*");

        // Type mapping: producer-side type id (short name) -> consumer-side class.
        // All services agree on the short names — they decouple from each other's packages.
        String mapping = "seatsReserved:" + SeatsReservedEvent.class.getName()
                + ",paymentCharged:" + PaymentChargedEvent.class.getName()
                + ",paymentFailed:" + PaymentFailedEvent.class.getName()
                + ",compensate:" + BookingCompensateEvent.class.getName();
        props.put(JsonDeserializer.TYPE_MAPPINGS, mapping);
        return new DefaultKafkaConsumerFactory<>(props);
    }

    @Bean
    public ConcurrentKafkaListenerContainerFactory<String, Object> sagaKafkaListenerContainerFactory() {
        ConcurrentKafkaListenerContainerFactory<String, Object> factory =
                new ConcurrentKafkaListenerContainerFactory<>();
        factory.setConsumerFactory(sagaConsumerFactory());
        // Retry 3 times with 1s backoff, then move on (in real life: send to a DLQ topic).
        factory.setCommonErrorHandler(new DefaultErrorHandler(new FixedBackOff(1_000L, 3)));
        return factory;
    }
}
