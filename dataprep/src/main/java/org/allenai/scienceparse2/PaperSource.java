package org.allenai.scienceparse2;

import java.io.IOException;
import java.io.InputStream;

/**
 * Encapsulates a way to get PDFs from paper ids.
 */
public interface PaperSource {
    InputStream getPdf(String paperId) throws IOException;
}
