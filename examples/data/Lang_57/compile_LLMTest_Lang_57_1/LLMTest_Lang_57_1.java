import org.junit.*;
import org.junit.Test;
import static org.junit.Assert.*;
import static org.junit.Assume.*;
import org.apache.commons.lang.LocaleUtils;
import static org.apache.commons.lang.LocaleUtils.*;
import java.math.BigDecimal;
import java.math.BigInteger;
import java.util.*;

public class LLMTest_Lang_57_1 {

@Test
public void testIsAvailableLocale() {
        LocaleUtils localeUtils = new LocaleUtils();
        Locale locale = new Locale("en", "US");
        boolean result = localeUtils.isAvailableLocale(locale);
        assertTrue(result);
    }

}