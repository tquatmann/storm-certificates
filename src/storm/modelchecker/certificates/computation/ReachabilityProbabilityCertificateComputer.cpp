#include "storm/modelchecker/certificates/computation/ReachabilityProbabilityCertificateComputer.h"

#include <vector>

#include "storm/adapters/RationalNumberAdapter.h"
#include "storm/environment/solver/EigenSolverEnvironment.h"
#include "storm/environment/solver/MinMaxSolverEnvironment.h"
#include "storm/exceptions/UnmetRequirementException.h"
#include "storm/modelchecker/certificates/ReachabilityProbabilityCertificate.h"
#include "storm/modelchecker/reachability/ReachabilityProbabilityToRewardTransformer.h"
#include "storm/solver/IterativeMinMaxLinearEquationSolver.h"
#include "storm/solver/LinearEquationSolver.h"
#include "storm/solver/MinMaxLinearEquationSolver.h"
#include "storm/solver/helper/IntervalterationHelper.h"
#include "storm/solver/helper/ValueIterationOperator.h"
#include "storm/storage/BitVector.h"
#include "storm/storage/SparseMatrix.h"
#include "storm/storage/StronglyConnectedComponentDecomposition.h"
#include "storm/utility/Extremum.h"
#include "storm/utility/NumberTraits.h"
#include "storm/utility/OptionalRef.h"
#include "storm/utility/Stopwatch.h"
#include "storm/utility/vector.h"

#include "storm/settings/SettingsManager.h"                // TODO: get rid of this
#include "storm/settings/modules/CertificationSettings.h"  // TODO: get rid of this

namespace storm::modelchecker {

template<typename ValueType, bool Nondeterministic, storm::OptimizationDirection Dir>
class LowerUpperValueCertificateComputer {
   public:
    enum class Algorithm { FpII, ExPI };

    template<typename VT, bool SingleValue>
    struct SubsystemData {
        storm::storage::SparseMatrix<VT> transitions;
        storm::storage::BitVector exitChoices;
        using ValueVectorType = std::conditional_t<SingleValue, std::vector<VT>, std::pair<std::vector<VT>, std::vector<VT>>>;
        ValueVectorType offsets;
        ValueVectorType operands;
    };

    LowerUpperValueCertificateComputer(storm::storage::SparseMatrix<ValueType> const& transitionProbabilityMatrix,
                                       storm::storage::BitVector const& terminalStates, storm::storage::BitVector const& prob1States)
        : transitionProbabilityMatrix(transitionProbabilityMatrix), terminalStates(terminalStates), prob1States(prob1States) {}

    Algorithm getAlgorithm() const {
        auto const& certSettings = storm::settings::getModule<storm::settings::modules::CertificationSettings>();

        auto algStr = certSettings.getMethod();
        if (algStr == "fp-ii") {
            return Algorithm::FpII;
        } else {
            assert(algStr == "ex-pi");
            return Algorithm::ExPI;
        }
    }

    std::pair<std::vector<ValueType>, std::vector<ValueType>> compute(bool relative, ValueType const& precision) {
        auto const& certSettings = storm::settings::getModule<storm::settings::modules::CertificationSettings>();
        auto alg = getAlgorithm();
        initializeValueVectors(alg);
        // TODO: Mec decomposition?
        // TODO: Choices tracking?
        if (certSettings.isUseTopologicalSet()) {
            computeTopological(alg, relative, precision, ~terminalStates);
        } else {
            computeForSubsystem(alg, relative, precision, ~terminalStates);
        }
        if (globalUpperValues.empty()) {
            // Exact methods do not populate the upper values.
            globalUpperValues = globalLowerValues;  // copy lower values
        }
        return {std::move(globalLowerValues), std::move(globalUpperValues)};
    }

   private:
    void initializeValueVectors(Algorithm alg) {
        globalLowerValues.assign(terminalStates.size(), storm::utility::zero<ValueType>());
        storm::utility::vector::setVectorValues(globalLowerValues, prob1States, storm::utility::one<ValueType>());

        if (alg == Algorithm::FpII) {
            globalUpperValues.assign(terminalStates.size(), storm::utility::one<ValueType>());
            auto prob0States = terminalStates ^ prob1States;
            storm::utility::vector::setVectorValues(globalUpperValues, prob0States, storm::utility::zero<ValueType>());
        } else {
            STORM_LOG_ASSERT(alg == Algorithm::ExPI, "Unsupported algorithm.");
            globalUpperValues.clear();  // Do not use upper values for this.
        }
    }

    storm::storage::BitVector getSubsystemExitChoices(storm::storage::BitVector const& subsystem, uint64_t numSubystemRows) const {
        storm::storage::BitVector result(numSubystemRows, false);
        uint64_t currSubsystemRow = 0;
        for (auto const state : subsystem) {
            for (auto const rowIndex : transitionProbabilityMatrix.getRowGroupIndices(state)) {
                auto row = transitionProbabilityMatrix.getRow(rowIndex);
                if (std::any_of(row.begin(), row.end(), [&subsystem](auto const& entry) { return !subsystem.get(entry.getColumn()); })) {
                    result.set(currSubsystemRow);
                }
                ++currSubsystemRow;
            }
        }
        return result;
    }

    template<typename VT>
    std::vector<VT> getSubsystemExitValues(storm::storage::BitVector const& subsystem, storm::storage::BitVector const& subsystemExitChoices,
                                           std::vector<ValueType> const& globalValues) const {
        std::vector<VT> result;
        result.reserve(subsystemExitChoices.size());
        for (auto const state : subsystem) {
            for (auto const rowIndex : transitionProbabilityMatrix.getRowGroupIndices(state)) {
                auto rowValue = storm::utility::zero<VT>();
                if (subsystemExitChoices.get(result.size())) {
                    for (auto const& entry : transitionProbabilityMatrix.getRow(rowIndex)) {
                        if (!subsystem.get(entry.getColumn())) {
                            rowValue += storm::utility::convertNumber<VT, ValueType>(entry.getValue() * globalValues[entry.getColumn()]);
                        }
                    }
                }
                result.push_back(std::move(rowValue));
            }
        }
        result.shrink_to_fit();
        STORM_LOG_ASSERT(result.size() == subsystemExitChoices.size(), "Unexpected number of rows in subsystem.");
        return result;
    }

    template<typename VT>
    storm::storage::SparseMatrix<VT> getSubsystemMatrix(storm::storage::BitVector const& subsystem) const {
        auto subMatrix = transitionProbabilityMatrix.getSubmatrix(true, subsystem, subsystem);
        if constexpr (std::is_same_v<VT, ValueType>) {
            return subMatrix;
        } else {
            return subMatrix.template toValueType<VT>();
        }
    }

    template<typename VT>
    std::vector<VT> getSubsystemOperands(storm::storage::BitVector const& subsystem, uint64_t numSubystemRowGroups,
                                         std::vector<ValueType> const& globalValues) const {
        std::vector<VT> result;
        result.reserve(numSubystemRowGroups);
        for (uint64_t i : subsystem) {
            result.push_back(storm::utility::convertNumber<VT>(globalValues[i]));
        }
        return result;
    }

    template<typename VT, bool SingleValue>
    SubsystemData<VT, SingleValue> initializeSubsystemData(storm::storage::BitVector const& subsystem) const {
        auto result = SubsystemData<VT, SingleValue>{getSubsystemMatrix<VT>(subsystem), {}, {}};
        result.exitChoices = getSubsystemExitChoices(subsystem, result.transitions.getRowCount());
        if constexpr (SingleValue) {
            STORM_LOG_ASSERT(globalUpperValues.empty(), "Expected upper values not to be initialized.");
            result.offsets = getSubsystemExitValues<VT>(subsystem, result.exitChoices, globalLowerValues);
            result.operands = getSubsystemOperands<VT>(subsystem, result.transitions.getRowCount(), globalLowerValues);
        } else {
            STORM_LOG_ASSERT(!globalUpperValues.empty(), "Expected upper values to be initialized.");
            result.offsets = {getSubsystemExitValues<VT>(subsystem, result.exitChoices, globalLowerValues),
                              getSubsystemExitValues<VT>(subsystem, result.exitChoices, globalUpperValues)};
            result.operands = {getSubsystemOperands<VT>(subsystem, result.transitions.getRowCount(), globalLowerValues),
                               getSubsystemOperands<VT>(subsystem, result.transitions.getRowCount(), globalUpperValues)};
        }
        return result;
    }

    template<typename VT>
    void setGlobalValuesFromSubsystem(storm::storage::BitVector const& subsystem, std::vector<VT>& localValues, std::vector<ValueType>& globalValues) {
        STORM_LOG_ASSERT(subsystem.getNumberOfSetBits() == localValues.size(), "Unexpected number of values in subsystem.");
        STORM_LOG_ASSERT(subsystem.size() == globalValues.size(), "Unexpected number of values.");
        auto localValuesIt = localValues.cbegin();
        for (auto i : subsystem) {
            globalValues[i] = storm::utility::convertNumber<ValueType>(*localValuesIt);
            ++localValuesIt;
        }
    }

    void computeForSingleStateSubsystem(uint64_t state) {
        storm::utility::Extremum<Dir, ValueType> lowerValue, upperValue;
        bool const computeUpperValues = !globalUpperValues.empty();
        for (auto const rowIndex : transitionProbabilityMatrix.getRowGroupIndices(state)) {
            auto lowerRowValue = storm::utility::zero<ValueType>();
            auto upperRowValue = storm::utility::zero<ValueType>();
            std::optional<ValueType> diagonalEntry;
            for (auto const& entry : transitionProbabilityMatrix.getRow(rowIndex)) {
                if (state == entry.getColumn()) {
                    STORM_LOG_ASSERT(!diagonalEntry, "Already have diag entry.");
                    diagonalEntry = entry.getValue();
                } else {
                    lowerRowValue += entry.getValue() * globalLowerValues[entry.getColumn()];
                    if (computeUpperValues) {
                        upperRowValue += entry.getValue() * globalUpperValues[entry.getColumn()];
                    }
                }
            }
            if (diagonalEntry.has_value()) {
                if (diagonalEntry.value() == storm::utility::one<ValueType>()) {
                    // This choice is a self-loop and can therefore never be optimal
                    continue;
                }
                lowerRowValue /= storm::utility::one<ValueType>() - diagonalEntry.value();
                if (computeUpperValues) {
                    upperRowValue /= storm::utility::one<ValueType>() - diagonalEntry.value();
                }
            }
            lowerValue &= lowerRowValue;
            if (computeUpperValues) {
                upperValue &= upperRowValue;
            }
        }
        STORM_LOG_ASSERT(!lowerValue.empty() && !(computeUpperValues && upperValue.empty()),
                         "Failed to compute lower and/or upper value for state " << state << ".");
        globalLowerValues[state] = *lowerValue;
        if (computeUpperValues) {
            globalUpperValues[state] = *upperValue;
        }
    }

    void computeForSubsystemFpII(bool relative, ValueType const& precision, storm::storage::BitVector const& subsystemStates) {
        using VT = std::conditional_t<storm::NumberTraits<ValueType>::IsExact, double, ValueType>;  // VT is imprecise
        std::optional<storm::OptimizationDirection> constexpr optionalDir = Nondeterministic ? std::optional<storm::OptimizationDirection>(Dir) : std::nullopt;

        auto subsystemData = initializeSubsystemData<VT, false>(subsystemStates);

        auto viOp = std::make_shared<storm::solver::helper::ValueIterationOperator<VT, !Nondeterministic>>();
        viOp->setMatrixBackwards(subsystemData.transitions);
        storm::solver::helper::IntervalIterationHelper<VT, !Nondeterministic> iiHelper(viOp);
        uint64_t numIterations = 0;
        iiHelper.II(subsystemData.operands, subsystemData.offsets, numIterations, relative, storm::utility::convertNumber<VT>(precision),
                    optionalDir);  // TODO: relevant values?
        setGlobalValuesFromSubsystem<VT>(subsystemStates, subsystemData.operands.first, globalLowerValues);
        setGlobalValuesFromSubsystem<VT>(subsystemStates, subsystemData.operands.second, globalUpperValues);
    }

    void computeForSubsystemExPI(storm::storage::BitVector const& subsystemStates) {
        using VT = std::conditional_t<storm::NumberTraits<ValueType>::IsExact, ValueType, storm::RationalNumber>;  // VT is exact
        std::optional<storm::OptimizationDirection> constexpr optionalDir = Nondeterministic ? std::optional<storm::OptimizationDirection>(Dir) : std::nullopt;

        auto subsystemData = initializeSubsystemData<VT, true>(subsystemStates);

        storm::Environment env;
        env.solver().minMax().setMethod(storm::solver::MinMaxMethod::ViToPi);
        env.solver().setLinearEquationSolverType(storm::solver::EquationSolverType::Eigen);
        env.solver().eigen().setMethod(solver::EigenLinearEquationSolverMethod::SparseLU);
        env.solver().setForceExact(true);

        if constexpr (Nondeterministic) {
            auto solverFactory = storm::solver::GeneralMinMaxLinearEquationSolverFactory<VT>();
            auto req = solverFactory.getRequirements(env, true, false, Dir, false, false);
            std::vector<uint64_t> initSched;
            if (req.validInitialScheduler()) {
                storm::storage::SparseMatrix<VT> const& submatrix = subsystemData.transitions;
                auto exitStates = submatrix.getRowGroupFilter(subsystemData.exitChoices, false);
                storm::storage::Scheduler<VT> validScheduler(exitStates.size());
                storm::utility::graph::computeSchedulerProb1E(storm::storage::BitVector(exitStates.size(), true), submatrix, submatrix.transpose(true),
                                                              ~exitStates, exitStates, validScheduler);

                // Extract the relevant parts of the scheduler for the solver.
                initSched.reserve(exitStates.size());
                for (uint64_t i = 0; i < exitStates.size(); ++i) {
                    if (exitStates.get(i)) {
                        initSched.push_back(validScheduler.getChoice(i).getDeterministicChoice());
                    } else {
                        auto const groupStart = submatrix.getRowGroupIndices()[i];
                        auto globalChoice = subsystemData.exitChoices.getNextSetIndex(groupStart);
                        STORM_LOG_ASSERT(globalChoice < submatrix.getRowGroupIndices()[i + 1], "No valid choice for exit state " << i << ".");
                        initSched.push_back(globalChoice - groupStart);
                    }
                }
                req.clearValidInitialScheduler();
            }
            STORM_LOG_THROW(!req.hasEnabledCriticalRequirement(), storm::exceptions::UnmetRequirementException,
                            "Unmet requirements for MinMax linear equation solver: " << req.getEnabledRequirementsAsString() << ".");
            auto solver = solverFactory.create(env, std::move(subsystemData.transitions));
            solver->setHasUniqueSolution(true);  // TODO: Has no end components?
            solver->setInitialScheduler(std::move(initSched));
            solver->setRequirementsChecked(true);
            // TODO: relevant values?
            solver->solveEquations(env, Dir, subsystemData.operands, subsystemData.offsets);
        } else {
            auto solverFactory = storm::solver::GeneralLinearEquationSolverFactory<VT>();
            if (solverFactory.getEquationProblemFormat(env) == storm::solver::LinearEquationSolverProblemFormat::EquationSystem) {
                subsystemData.transitions = storm::storage::SparseMatrix<VT>(subsystemData.transitions, true);  // insert diag entries
                subsystemData.transitions.convertToEquationSystem();  // Note: Invalidates uses of this matrix as 'transitions'
            }
            auto req = solverFactory.getRequirements(env);
            STORM_LOG_THROW(!req.hasEnabledCriticalRequirement(), storm::exceptions::UnmetRequirementException,
                            "Unmet requirements for linear equation solver: " << req.getEnabledRequirementsAsString() << ".");
            auto solver = solverFactory.create(env, std::move(subsystemData.transitions));
            // TODO: relevant values?
            solver->solveEquations(env, subsystemData.operands, subsystemData.offsets);
        }
        // TODO: handle the fact that we don't need to handle lower/upper bounds separately
        setGlobalValuesFromSubsystem<VT>(subsystemStates, subsystemData.operands, globalLowerValues);
        STORM_LOG_ASSERT(globalUpperValues.empty(), "Expected upper values not to be initialized.");
    }

    void computeForSubsystem(Algorithm alg, bool relative, ValueType const& precision, storm::storage::BitVector const& subsystemStates) {
        switch (alg) {
            case Algorithm::FpII:
                computeForSubsystemFpII(relative, precision, subsystemStates);
                break;
            case Algorithm::ExPI:
                computeForSubsystemExPI(subsystemStates);
                break;
            default:
                STORM_LOG_ASSERT(false, "Unsupported algorithm.");
        }
    }

    void computeTopological(Algorithm alg, bool relative, ValueType const& precision, storm::storage::BitVector const& nonTerminalStates) {
        STORM_LOG_TRACE("Creating SCC decomposition.");
        storm::utility::Stopwatch sccSw(true);
        storm::storage::StronglyConnectedComponentDecomposition<ValueType> const sccDecomposition(
            transitionProbabilityMatrix,
            storm::storage::StronglyConnectedComponentDecompositionOptions().forceTopologicalSort().computeSccDepths().subsystem(nonTerminalStates));
        uint64_t const longestSccChainSize = sccDecomposition.getMaxSccDepth() + 1;
        sccSw.stop();
        STORM_LOG_INFO("SCC decomposition computed in "
                       << sccSw << ". Found " << sccDecomposition.size() << " SCC(s) containing a total of " << nonTerminalStates.getNumberOfSetBits()
                       << " states. Average SCC size is "
                       << static_cast<double>(transitionProbabilityMatrix.getRowGroupCount()) / static_cast<double>(sccDecomposition.size())
                       << ". Longest SCC chain size is " << longestSccChainSize << ".");
        if (sccDecomposition.size() == 1) {
            // Handle the case where there is just one large scc
            computeForSubsystem(alg, relative, precision, nonTerminalStates);
            return;
        }
        // At this point, solve each SCC individually
        storm::storage::BitVector sccAsBitvector(transitionProbabilityMatrix.getRowGroupCount(), false);  // allocate memory outside of loop
        for (uint64_t sccIndex = 0; sccIndex < sccDecomposition.size(); ++sccIndex) {
            auto const& scc = sccDecomposition.getBlock(sccIndex);
            if (scc.size() == 1) {
                // Directly solve single-state SCCs
                computeForSingleStateSubsystem(*scc.begin());
                continue;
            }
            sccAsBitvector.set(scc.begin(), scc.end());
            ValueType const sccPrecision =
                precision / storm::utility::convertNumber<ValueType, uint64_t>(longestSccChainSize - sccDecomposition.getSccDepth(sccIndex));
            computeForSubsystem(alg, relative, sccPrecision, sccAsBitvector);
            sccAsBitvector.clear();
        }
    }

    storm::storage::SparseMatrix<ValueType> const& transitionProbabilityMatrix;
    storm::storage::BitVector const& terminalStates;
    storm::storage::BitVector const& prob1States;
    std::vector<ValueType> globalLowerValues, globalUpperValues;
    storm::OptionalRef<storm::storage::BitVector> globalPossibleChoices;
};

template<typename ValueType, storm::OptimizationDirection Dir>
std::vector<typename ReachabilityProbabilityCertificate<ValueType>::RankingType> computeLowerBoundRanking(
    storm::storage::SparseMatrix<ValueType> const& transitionProbabilityMatrix, storm::storage::SparseMatrix<ValueType> const& backwardTransitions,
    storm::storage::BitVector const& targetStates, storm::OptionalRef<storm::storage::BitVector const> constraintStates,
    storm::OptionalRef<storm::storage::BitVector const> choiceConstraint = storm::NullRef) {
    using RankingType = typename ReachabilityProbabilityCertificate<ValueType>::RankingType;
    RankingType const InfRank = ReachabilityProbabilityCertificate<ValueType>::InfRank;
    auto const& rowGroupIndices = transitionProbabilityMatrix.getRowGroupIndices();

    std::vector<RankingType> ranks(targetStates.size(), InfRank);
    storm::utility::vector::setVectorValues<RankingType>(ranks, targetStates, 0);
    std::vector<uint64_t> stateQueue(targetStates.begin(), targetStates.end());
    while (!stateQueue.empty()) {
        auto const state = stateQueue.back();
        stateQueue.pop_back();
        for (auto const& preEntry : backwardTransitions.getRow(state)) {
            auto const preState = preEntry.getColumn();
            if (ranks[preState] != InfRank) {
                continue;
            }
            if (constraintStates.has_value() && !constraintStates->get(preState)) {
                continue;
            }
            storm::utility::Extremum<storm::solver::invert(Dir), RankingType> newRankMinusOne;
            // Iterate over all (selected) choices of the pre-state
            uint64_t currentChoice = rowGroupIndices[preState];
            if (choiceConstraint.has_value()) {
                currentChoice = choiceConstraint->getNextSetIndex(currentChoice);
            }
            auto nextChoice = [&choiceConstraint](auto c) { return choiceConstraint.has_value() ? choiceConstraint->getNextSetIndex(c + 1) : c + 1; };
            for (auto const endChoice = rowGroupIndices[preState + 1]; currentChoice < endChoice; currentChoice = nextChoice(currentChoice)) {
                // Compute current rank for this choice
                storm::utility::Minimum<RankingType> choiceRank;
                for (auto const& entry : transitionProbabilityMatrix.getRow(currentChoice)) {
                    if (storm::utility::isZero(entry.getValue())) {
                        continue;
                    }
                    choiceRank &= ranks[entry.getColumn()];
                }
                STORM_LOG_ASSERT(!choiceRank.empty(),
                                 "Ranking operator failed to compute rank for state " << state << " since the row " << choiceConstraint << " is empty.");
                newRankMinusOne &= *choiceRank;
            }
            STORM_LOG_ASSERT(!newRankMinusOne.empty(), "Ranking operator failed to compute rank for state " << state << ".");
            if (*newRankMinusOne != InfRank) {
                ranks[preState] = *newRankMinusOne + 1;
                stateQueue.push_back(preState);
            }
        }
    }
    return ranks;
}

template<typename ValueType>
struct CertificateData {
    std::vector<ValueType> lowerValues, upperValues;
    std::vector<typename ReachabilityProbabilityCertificate<ValueType>::RankingType> ranks;
};

template<typename ValueType, bool Nondeterministic, storm::OptimizationDirection Dir = storm::OptimizationDirection::Minimize>
CertificateData<ValueType> computeReachabilityProbabilityCertificateData(storm::Environment const& env,
                                                                         storm::storage::SparseMatrix<ValueType> const& transitionProbabilityMatrix,
                                                                         storm::storage::BitVector const& targetStates,
                                                                         storm::OptionalRef<storm::storage::BitVector const> constraintStates) {
    std::optional<storm::OptimizationDirection> constexpr optionalDir = Nondeterministic ? std::optional<storm::OptimizationDirection>(Dir) : std::nullopt;
    utility::BackwardTransitionCache<ValueType> backwardTransitionCache(transitionProbabilityMatrix);
    auto toRewardData = ReachabilityProbabilityToRewardTransformer<ValueType>(transitionProbabilityMatrix, backwardTransitionCache)
                            .transform(optionalDir, targetStates, constraintStates);
    STORM_LOG_ASSERT(toRewardData.terminalStateValues.size() == 1 && toRewardData.terminalStateValues.front().first.isOne(),
                     "Expected exactly one terminal state value with value 1.");
    LowerUpperValueCertificateComputer<ValueType, Nondeterministic, Dir> computer(transitionProbabilityMatrix, toRewardData.terminalStates,
                                                                                  toRewardData.terminalStateValues.front().second);

    auto [lowerValues, upperValues] = computer.compute(env.solver().minMax().getRelativeTerminationCriterion(),
                                                       storm::utility::convertNumber<ValueType>(env.solver().minMax().getPrecision()));
    auto ranks = computeLowerBoundRanking<ValueType, Dir>(transitionProbabilityMatrix, backwardTransitionCache.get(), targetStates,
                                                          constraintStates);  // TODO: choice constraints for max reach
    return {std::move(lowerValues), std::move(upperValues), std::move(ranks)};
}

template<typename ValueType>
CertificateData<ValueType> computeReachabilityProbabilityCertificateData(storm::Environment const& env, std::optional<storm::OptimizationDirection> dir,
                                                                         storm::storage::SparseMatrix<ValueType> const& transitionProbabilityMatrix,
                                                                         storm::storage::BitVector const& targetStates,
                                                                         storm::OptionalRef<storm::storage::BitVector const> constraintStates) {
    CertificateData<ValueType> data;
    if (dir.has_value()) {
        if (storm::solver::maximize(*dir)) {
            data = computeReachabilityProbabilityCertificateData<ValueType, true, storm::OptimizationDirection::Maximize>(env, transitionProbabilityMatrix,
                                                                                                                          targetStates, constraintStates);
        } else {
            data = computeReachabilityProbabilityCertificateData<ValueType, true, storm::OptimizationDirection::Minimize>(env, transitionProbabilityMatrix,
                                                                                                                          targetStates, constraintStates);
        }
    } else {
        data = computeReachabilityProbabilityCertificateData<ValueType, false>(env, transitionProbabilityMatrix, targetStates, constraintStates);
    }
    return data;
}

template<typename ValueType>
std::unique_ptr<ReachabilityProbabilityCertificate<ValueType>> computeReachabilityProbabilityCertificate(
    storm::Environment const& env, std::optional<storm::OptimizationDirection> dir, storm::storage::SparseMatrix<ValueType> const& transitionProbabilityMatrix,
    storm::storage::BitVector targetStates, std::optional<storm::storage::BitVector> constraintStates, std::string targetLabel,
    std::optional<std::string> constraintLabel) {
    CertificateData<ValueType> data;
    std::unique_ptr<ReachabilityProbabilityCertificate<ValueType>> result;
    if (constraintStates.has_value()) {
        data = computeReachabilityProbabilityCertificateData<ValueType>(env, dir, transitionProbabilityMatrix, targetStates, constraintStates.value());
        result = std::make_unique<ReachabilityProbabilityCertificate<ValueType>>(dir, std::move(targetStates), std::move(constraintStates.value()),
                                                                                 std::move(targetLabel), constraintLabel.value_or("constraint"));
    } else {
        data = computeReachabilityProbabilityCertificateData<ValueType>(env, dir, transitionProbabilityMatrix, targetStates, storm::NullRef);
        result = std::make_unique<ReachabilityProbabilityCertificate<ValueType>>(dir, std::move(targetStates), std::move(targetLabel));
    }
    result->setLowerBoundsCertificate(std::move(data.lowerValues), std::move(data.ranks));
    result->setUpperBoundsCertificate(std::move(data.upperValues));
    return result;
}

template std::unique_ptr<ReachabilityProbabilityCertificate<double>> computeReachabilityProbabilityCertificate<double>(
    storm::Environment const& env, std::optional<storm::OptimizationDirection> dir, storm::storage::SparseMatrix<double> const& transitionProbabilityMatrix,
    storm::storage::BitVector targetStates, std::optional<storm::storage::BitVector> constraintStates, std::string targetLabel,
    std::optional<std::string> constraintLabel);
template std::unique_ptr<ReachabilityProbabilityCertificate<storm::RationalNumber>> computeReachabilityProbabilityCertificate<storm::RationalNumber>(
    storm::Environment const& env, std::optional<storm::OptimizationDirection> dir,
    storm::storage::SparseMatrix<storm::RationalNumber> const& transitionProbabilityMatrix, storm::storage::BitVector targetStates,
    std::optional<storm::storage::BitVector> constraintStates, std::string targetLabel, std::optional<std::string> constraintLabel);

}  // namespace storm::modelchecker