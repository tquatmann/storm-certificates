#include "storm/settings/modules/CertificationSettings.h"

#include "storm/settings/ArgumentBuilder.h"
#include "storm/settings/Option.h"
#include "storm/settings/OptionBuilder.h"

namespace storm::settings::modules {

const std::string CertificationSettings::moduleName = "cert";

const std::string produceCertificateOption = "certificate";
const std::string useTopologicalOption = "topological";
const std::string methodOption = "method";

CertificationSettings::CertificationSettings() : ModuleSettings(moduleName) {
    this->addOption(
        storm::settings::OptionBuilder(moduleName, produceCertificateOption, false, "If set, a certificate will be produced (if supported).").build());

    this->addOption(
        storm::settings::OptionBuilder(moduleName, useTopologicalOption, false, "If set, uses topological methods to produce certificates.").build());
    std::vector<std::string> const methodNames = {"fp-ii", "ex-pi"};
    this->addOption(storm::settings::OptionBuilder(moduleName, methodOption, true, "The method to use for certification.")
                        .addArgument(storm::settings::ArgumentBuilder::createStringArgument("name", "The name of the method to use.")
                                         .setDefaultValueString("fp-ii")
                                         .addValidatorString(ArgumentValidatorFactory::createMultipleChoiceValidator(methodNames))
                                         .build())
                        .build());
}

bool CertificationSettings::isProduceCertificateSet() const {
    return this->getOption(produceCertificateOption).getHasOptionBeenSet();
}

bool CertificationSettings::isUseTopologicalSet() const {
    return this->getOption(useTopologicalOption).getHasOptionBeenSet();
}

std::string CertificationSettings::getMethod() const {
    // TODO: make this an enum
    return this->getOption(methodOption).getArgumentByName("name").getValueAsString();
}

bool CertificationSettings::check() const {
    return true;
}

}  // namespace storm::settings::modules
