/*  Sprite Identifier for Pokemon
 *
 *  From: https://github.com/PokemonAutomation/Arduino-Source
 *
 *
 *      Identify a Pokemon sprite by comparing it to stored sprites.
 *      TODO: The current method blocks until all the comparisons are complete.
 *      If the computation takes a significant amount of time, it might make
 *      sense to start a worker thread if the inference doesn't already have its
 *      own thread.
 *
 */

#include "opencv2/imgproc.hpp"  // TODO: needs replacing with the actual path.

#include "CommonFramework/Inference/ImageTools.h"
#include "PokemonSwSh_SpriteIdentifierPokemon.h"

namespace PokemonAutomation{
namespace NintendoSwitch{
namespace PokemonSwSh{


SpriteIdentifierPokemon::SpriteIdentifierPokemon(
    VideoFeed& feed, Logger& logger
)
    : m_feed(feed)
    , m_logger(logger)
{}

SpriteIdentifierPokemon::identify_pokemon(
    cv::Mat image,
    sprite_with_id& sprites_to_match,
    double scale=1,
    bool is_silhouette=false
)
{
    /**
     * @brief Identify a Pokemon sprite in an image.
     *
     * @param image OpenCV image of the Pokemon to be identified (BGRA format)
     * @param sprites_to_match Array of sprite-ID pairs used as a reference for
     * matching. Note that the image format should be BGRA which is the format
     * returned when reading a PNG with the flag cv::IMREAD_UNCHANGED.
     * @param scale Scale of the sprite to be identified in the image. If the
     * sprite is double the stored sprite size, the scale would be 2.
     * @param is_silhouette If the image contains a black silhouette of the
     * Pokemon then this parameter should be set to true.
     * @return ID number of the identified Pokemon.
     */
    
    // Resize the image as needed based on the scale of the sprite in the image.
    if(scale != 1)
    {
        // The image is a different size than the sprite, so scale the image
        // accordingly.
        double resize_factor = (double) 1 / scale;
        // Note: the cv::INTER_NEAREST flag is needed to ensure that the image
        // is not interpolated during rescaling.
        cv::resize(
            image, image, cv::Size(), resize_factor, resize_factor,
            cv::INTER_NEAREST
        );
    }

    // Iterate over the list of potential sprites to find a match
    int result_cols; int result_rows;
    cv::Mat result; cv::Mat mask;
    double minVal; double maxVal; cv::Point minLoc; cv::Point maxLoc;
    match_result match = {0, 0};
    for(sprite_with_id &sprite_to_match: sprites_to_match)
    {
        // Make a deep copy of the sprite image so modifying it doesn't affect
        // the original image.
        // TODO: I'm not sure whether making a copy of every sprite will slow
        // down computation too much. If it's an issue, you could probably
        // reload the sprite assets after every identification or pre-generate
        // the sprite silhouettes.
        cv::Mat sprite_image_copy = sprite_to_match.sprite_image.clone();

        // Create a mask from the transparent channel of the sprite.
        cv::threshold(
            sprite_image_copy.at(3), mask, 255, 255,
            cv::THRESHOLD_BINARY_INVERTED
        );

        // If the image contains a silhouette, use a black template image.
        if(is_silhouette)
        {
            // The mask takes care of the shape already so we just have to set
            // all the channels of the image to zero.
            sprite_image_copy = cv::Scalar(0, 0, 0, 0);
        }

        // Create a result matrix that stores the match score of the template
        // at every possible position within the image.
        result_cols = image.cols - sprite_image_copy.cols + 1;
        result_rows = image.rows - sprite_image_copy.rows + 1;
        result.create(result_rows, result_cols, cv::CV_32FC1);

        // Run the template matching sequence and isolate the best result
        cv::matchTemplate(
            image, sprite_image_copy, result, cv::TM_CCORR_NORMED,
            mask
        );
        cv::minMaxLoc(result, &minVal, &maxVal, &minLoc, &maxLoc, cv::Mat());

        // Store the best match if it is better than the current best.
        if(maxVal > match.best_match_value)
        {
            match.best_match_value = maxVal;
            match.best_match_id = sprite_to_match.id_number;
        }
    }

    return match;
}

SpriteIdentifierPokemon::identify_pokemon(
    const QImage& image,
    sprite_with_id& sprites_to_match,
    double scale=1,
    bool is_silhouette=false
)
{
    /**
     * @brief Identify a Pokemon sprite in an image.
     * Overloaded version that first converts a QImage to the required OpenCV
     * format.
     *
     * @param image Qt image of the Pokemon to be identified (ARGB format)
     * @param sprites_to_match Array of sprite-ID pairs used as a reference for
     * matching. Note that the image format should be ARGB.
     * @param scale Scale of the sprite to be identified in the image. If the
     * sprite is double the stored sprite size, the scale would be 2.
     * @param is_silhouette If the image contains a black silhouette of the
     * Pokemon then this parameter should be set to true.
     * @return ID number of the identified Pokemon.
     */

    // Convert the QImage to the OpenCV image format (Mat).
    // Make a copy so that changing the image (for example scaling) won't affect
    // the original.
    // TODO: check that image conversion actually works.
    cv::Mat converted_image = cv::Mat(
        image.height(), image.width(), cv::CV_32FC1, image.bits(),
        image.bytesPerLine()
    ).clone();
    cv::cvtColor(converted_image, converted_image, cv::CV_RGBA2BGRA);
    return SpriteIdentifierPokemon::identify_pokemon(
        converted_image, sprite_to_match, scale, is_silhouette);
}

SpriteIdentifierPokemon::identify_pokemon(
    InferenceBoxScope spritebox,
    sprite_with_id& sprites_to_match,
    double scale=1,
    bool is_silhouette=false
)
{
    /**
     * @brief Identify a Pokemon sprite in an image.
     * Overloaded version that first grabs a frame from the video feed and
     * analyzes a specified section of it.
     *
     * @param spritebox InferenceBoxScope object that specifies the part of the
     * video that will be assessed. Coordinates are fractional from 0 to 1.
     * @param sprites_to_match Array of sprite-ID pairs used as a reference for
     * matching. Note that the image format should be ARGB.
     * @param scale Scale of the sprite to be identified in the image. If the
     * sprite is double the stored sprite size, the scale would be 2.
     * @param is_silhouette If the image contains a black silhouette of the
     * Pokemon then this parameter should be set to true.
     * @return ID number of the identified Pokemon.
     */

    // Take a snapshot from the video feed.
    QImage snapshot = m_feed.snapshot();
    if (snapshot.isNull()){
        m_logger.log("RaidLobbyReader(): Screenshot failed.", "purple");
        return 0;
    }

    // Call the QImage version of the method with a QImage section.
    return SpriteIdentifierPokemon::identify_pokemon(
        extract_box(snapshot, spritebox),
        sprite_to_match,
        scale,
        is_silhouette
    );
}



}
}
}