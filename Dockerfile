FROM gcc:13

WORKDIR /app

COPY . .

RUN make inference

CMD ["./inference", "iris"]
