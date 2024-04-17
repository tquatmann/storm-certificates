#pragma once

#include <vector>
#include "storm/modelchecker/certificates/Certificate.h"
#include "storm/solver/OptimizationDirection.h"
#include "storm/storage/BitVector.h"

namespace storm {

class Environment;

namespace storage {

template<typename ValueType>
class SparseMatrix;

}

namespace modelchecker {

template<typename ValueType>
class ReachabilityProbabilityCertificate : public Certificate<ValueType> {
   public:
    using RankingType = uint64_t;
    static RankingType const InfRank = std::numeric_limits<RankingType>::max();
    ReachabilityProbabilityCertificate(std::optional<storm::OptimizationDirection> dir, storm::storage::BitVector targetStates,
                                       std::string targetLabel = "goal");
    virtual ~ReachabilityProbabilityCertificate() = default;
    virtual bool checkValidity(storm::models::Model<ValueType> const& model) const override;
    bool checkValidity(storm::storage::SparseMatrix<ValueType> const& transitionProbabilityMatrix) const;
    virtual storm::json<ValueType> toJson() const override;
    virtual std::string summaryString(storm::storage::BitVector const& relevantStates) const override;
    virtual std::unique_ptr<Certificate<ValueType>> clone() const override;

    void setLowerBoundsCertificate(std::vector<ValueType>&& values, std::vector<uint64_t>&& ranks);
    void setUpperBoundsCertificate(std::vector<ValueType>&& values);

    bool hasLowerBoundsCertificate() const;
    bool hasUpperBoundsCertificate() const;

   private:
    template<storm::OptimizationDirection Dir>
    bool checkValidityInternal(storm::storage::SparseMatrix<ValueType> const& transitionProbabilityMatrix) const;

    storm::storage::BitVector const targetStates;
    std::string const targetLabel;
    std::optional<storm::OptimizationDirection> const dir;

    struct {
        std::vector<ValueType> values;
    } upperBoundsCertificate;

    struct {
        std::vector<ValueType> values;
        std::vector<RankingType> ranks;
    } lowerBoundsCertificate;
};

}  // namespace modelchecker
}  // namespace storm