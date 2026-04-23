package com.bookmyshow.config;

import org.apache.kafka.clients.admin.NewTopic;
import org.springframework.context.annotation.Bean;
import org.springframework.context.annotation.Configuration;
import org.springframework.kafka.config.TopicBuilder;

@Configuration
public class KafkaTopicsConfig {

    public static final String BOOKING_CONFIRMED = "booking.confirmed";
    public static final String BOOKING_FAILED    = "booking.failed";

    @Bean
    public NewTopic bookingConfirmedTopic() {
        return TopicBuilder.name(BOOKING_CONFIRMED).partitions(3).replicas(1).build();
    }

    @Bean
    public NewTopic bookingFailedTopic() {
        return TopicBuilder.name(BOOKING_FAILED).partitions(3).replicas(1).build();
    }
}
