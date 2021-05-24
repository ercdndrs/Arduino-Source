/*  Sprite Identifier for Pokemon
 *
 *  From: https://github.com/PokemonAutomation/Arduino-Source
 *
 */

#ifndef PokemonAutomation_PokemonSwSh_SpriteIdentifierPokemon_H
#define PokemonAutomation_PokemonSwSh_SpriteIdentifierPokemon_H

#include "CommonFramework/Tools/Logger.h"
#include "CommonFramework/Tools/VideoFeed.h"
#include "CommonFramework/Tools/ProgramEnvironment.h"
#include "CommonFramework/Inference/VisualInferenceCallback.h"

namespace PokemonAutomation{
namespace NintendoSwitch{
namespace PokemonSwSh{

typedef unsigned short Pokemon_ID;  // TODO: Replace later.

struct sprite_with_id{
    Pokemon_ID id_number,
    Mat sprite_image
};

struct match_result{
    Pokemon_ID best_match_id,
    float best_match_value
};

class SpriteIdentifierPokemon : public VisualInferenceCallback{
public:
    SpriteIdentifierPokemon(VideoFeed& feed, Logger& logger);

    Pokemon_ID identify_pokemon(
        cv::Mat image,
        sprite_with_id& sprites_to_match,
        double scale=1,
        bool is_silhouette=false
    );

    Pokemon_ID identify_pokemon(
        const QImage& image,
        sprite_with_id& sprites_to_match,
        double scale=1,
        bool is_silhouette=false
    );

    Pokemon_ID identify_pokemon(
        InferenceBoxScope section,
        sprite_with_id& sprites_to_match,
        double scale=1,
        bool is_silhouette=false
    );

private:
    VideoFeed& m_feed;
    Logger& m_logger;
};

}
}
}

#endif