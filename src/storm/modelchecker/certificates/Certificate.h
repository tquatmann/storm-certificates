#pragma once

#include <memory>
#include <string>
#include "storm/adapters/JsonForward.h"
#include "storm/modelchecker/certificates/CertificateKind.h"

namespace storm {

namespace storage {
class BitVector;
}

namespace models {
template<typename ValueType>
class Model;
}

namespace modelchecker {

template<typename ValueType>
class Certificate {
   public:
    virtual ~Certificate() = default;
    virtual bool checkValidity(storm::models::Model<ValueType> const& model) const = 0;
    virtual storm::json<ValueType> toJson() const = 0;
    virtual std::string summaryString(storm::storage::BitVector const& relevantStates) const = 0;
    virtual std::unique_ptr<Certificate<ValueType>> clone() const = 0;

   protected:
    Certificate(CertificateKind kind) : kind(kind) {}

   private:
    CertificateKind const kind;
};

}  // namespace modelchecker
}  // namespace storm