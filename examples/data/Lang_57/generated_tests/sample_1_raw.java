@Test
public void testIsAvailableLocale() {
        LocaleUtils localeUtils = new LocaleUtils();
        Locale locale = new Locale("en", "US");
        boolean result = localeUtils.isAvailableLocale(locale);
        assertTrue(result);
    }
