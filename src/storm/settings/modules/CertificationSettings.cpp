#include "storm/settings/modules/CertificationSettings.h"

#include "storm/settings/ArgumentBuilder.h"
#include "storm/settings/Option.h"
#include "storm/settings/OptionBuilder.h"

namespace storm::settings::modules {

const std::string CertificationSettings::moduleName = "cert";

const std::string produceCertificateOption = "certificate";

CertificationSettings::CertificationSettings() : ModuleSettings(moduleName) {
    this->addOption(
        storm::settings::OptionBuilder(moduleName, produceCertificateOption, false, "If set, a certificate will be produced (if supported).").build());
}

bool CertificationSettings::isProduceCertificateSet() const {
    return this->getOption(produceCertificateOption).getHasOptionBeenSet();
}

bool CertificationSettings::check() const {
    return true;
}

}  // namespace storm::settings::modules
