package com.bookmyshow;

import org.springframework.boot.SpringApplication;
import org.springframework.boot.autoconfigure.SpringBootApplication;
import org.springframework.cache.annotation.EnableCaching;
import org.springframework.scheduling.annotation.EnableAsync;
import org.springframework.transaction.annotation.EnableTransactionManagement;

@SpringBootApplication
@EnableCaching                  // enables @Cacheable / @CacheEvict on services
@EnableAsync                    // enables @Async methods (runs on our custom executors)
@EnableTransactionManagement    // explicit for clarity (Spring Boot enables it by default)
public class BookMyShowApplication {
    public static void main(String[] args) {
        SpringApplication.run(BookMyShowApplication.class, args);
    }
}
