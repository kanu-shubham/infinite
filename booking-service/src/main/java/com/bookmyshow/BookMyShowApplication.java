package com.bookmyshow;

import org.springframework.boot.SpringApplication;
import org.springframework.boot.autoconfigure.SpringBootApplication;
import org.springframework.cache.annotation.EnableCaching;
import org.springframework.scheduling.annotation.EnableAsync;
import org.springframework.scheduling.annotation.EnableScheduling;
import org.springframework.transaction.annotation.EnableTransactionManagement;

@SpringBootApplication
@EnableCaching                  // enables @Cacheable / @CacheEvict on services
@EnableAsync                    // enables @Async methods (runs on our custom executors)
@EnableScheduling               // enables @Scheduled (OutboxPublisher runs every 500ms)
@EnableTransactionManagement    // explicit for clarity (Spring Boot enables it by default)
public class BookMyShowApplication {
    public static void main(String[] args) {
        SpringApplication.run(BookMyShowApplication.class, args);
    }
}
