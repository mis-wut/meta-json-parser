#ifndef META_JSON_PARSER_STACK_TOKEN_CUH
#define META_JSON_PARSER_STACK_TOKEN_CUH

namespace JsonParsers {
    struct StackToken {
        enum class Value : uint8_t {
            Object = 0x01,
            Array = 0x02,
            SeenValue = 0x10,
            SeenComma = 0x20,
            SeenKey = 0x40
        } value;

        static constexpr uint8_t SEEN_MASK = static_cast<uint8_t>(Value::SeenValue) |
                                             static_cast<uint8_t>(Value::SeenComma) |
                                             static_cast<uint8_t>(Value::SeenKey);

        __device__ __host__ __forceinline__ explicit operator Value() const { return value; }
        __device__ __host__ __forceinline__ bool operator==(Value v) const { return value == v; }
        __device__ __host__ __forceinline__ bool operator!=(Value v) const { return value != v; }

        __forceinline__ StackToken() = default;
        __device__ __forceinline__ StackToken(Value v) : value(v) { }

        [[nodiscard]] __device__ __host__ __forceinline__ bool IsObject() const {
            return static_cast<uint8_t>(value) & static_cast<uint8_t>(Value::Object);
        }
        [[nodiscard]] __device__ __host__ __forceinline__ bool IsArray() const {
            return static_cast<uint8_t>(value) & static_cast<uint8_t>(Value::Array);
        }
        [[nodiscard]] __device__ __host__ __forceinline__ bool NothingWasSeen() const {
            return !(static_cast<uint8_t>(value) & SEEN_MASK);
        }
        [[nodiscard]] __device__ __host__ __forceinline__ bool CommaWasSeen() const {
            return static_cast<uint8_t>(value) & static_cast<uint8_t>(Value::SeenComma);
        }
        [[nodiscard]] __device__ __host__ __forceinline__ bool ValueWasSeen() const {
            return static_cast<uint8_t>(value) & static_cast<uint8_t>(Value::SeenValue);
        }
        [[nodiscard]] __device__ __host__ __forceinline__ bool KeyWasSeen() const {
            return static_cast<uint8_t>(value) & static_cast<uint8_t>(Value::SeenKey);
        }
        [[nodiscard]] __device__ __host__ __forceinline__ bool CommaOrKeyWasSeen() const {
            return static_cast<uint8_t>(value) & (static_cast<uint8_t>(Value::SeenKey) | static_cast<uint8_t>(Value::SeenComma));
        }
        [[nodiscard]] __device__ __host__ __forceinline__ bool IsValuePossible() const {
            return (IsArray() && IsValuePossible_AssumeArray()) ||
                   (IsObject() && IsValuePossible_AssumeObject());
        }
        [[nodiscard]] __device__ __host__ __forceinline__ bool IsValuePossible_AssumeArray() const {
            return NothingWasSeen() || CommaWasSeen();
        }
        [[nodiscard]] __device__ __host__ __forceinline__ bool IsValuePossible_AssumeObject() const {
            return KeyWasSeen();
        }
        [[nodiscard]] __device__ __host__ __forceinline__ bool IsCommaPossible() const {
            return ValueWasSeen();
        }
        [[nodiscard]] __device__ __host__ __forceinline__ bool IsKeyPossible() const {
            return IsObject() && IsKeyPossible_AssumeObject();
        }
        [[nodiscard]] __device__ __host__ __forceinline__ bool IsKeyPossible_AssumeObject() const {
            return NothingWasSeen() || CommaWasSeen();
        }
        [[nodiscard]] __device__ __host__ __forceinline__ bool CanObjectEnd() const {
            return IsObject() && !CommaOrKeyWasSeen();
        }
        [[nodiscard]] __device__ __host__ __forceinline__ bool CanArrayEnd() const {
            return IsArray() && !CommaWasSeen();
        }

        __device__ __host__ __forceinline__ void SetSeenValue() {
            value = static_cast<Value>((static_cast<uint8_t>(value) & ~SEEN_MASK) | static_cast<uint8_t>(Value::SeenValue));
        }
        __device__ __host__ __forceinline__ void SetSeenComma() {
            value = static_cast<Value>((static_cast<uint8_t>(value) & ~SEEN_MASK) | static_cast<uint8_t>(Value::SeenComma));
        }
        __device__ __host__ __forceinline__ void SetSeenKey() {
            value = static_cast<Value>((static_cast<uint8_t>(value) & ~SEEN_MASK) | static_cast<uint8_t>(Value::SeenKey));
        }
        __device__ __host__ __forceinline__ void SetObject() {
            value = Value::Object;
        }
        __device__ __host__ __forceinline__ void SetArray() {
            value = Value::Array;
        }
    };
}

#endif //META_JSON_PARSER_STACK_TOKEN_CUH