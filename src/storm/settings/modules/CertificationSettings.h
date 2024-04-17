#pragma once

#include "storm/settings/modules/ModuleSettings.h"

namespace storm::settings::modules {

/*!
 * This class represents the settings for the native equation solver.
 */
class CertificationSettings : public ModuleSettings {
   public:
    CertificationSettings();

    /*!
     * Whether a certificate is to be produced
     */
    bool isProduceCertificateSet() const;

    bool check() const override;

    // The name of the module.
    static const std::string moduleName;
};

}  // namespace storm::settings::modules
